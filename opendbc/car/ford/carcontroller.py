import math
import numpy as np
from collections import deque
from cereal import log, messaging
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.lateral import ISO_LATERAL_ACCEL, apply_std_steer_angle_limits
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX
from selfdrive.modeld.constants import ModelConstants
from openpilot.common.params import Params

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

# CAN FD limits:
# Limit to average banked road since safety doesn't have the roll
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation. higher actual roll raises lateral acceleration
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~2.4 m/s^2


def anti_overshoot(apply_curvature, apply_curvature_last, v_ego):
  diff = 0.1
  tau = 5  # 5s smooths over the overshoot
  dt = DT_CTRL * CarControllerParams.STEER_STEP
  alpha = 1 - np.exp(-dt / tau)

  lataccel = apply_curvature * (v_ego ** 2)
  last_lataccel = apply_curvature_last * (v_ego ** 2)
  last_lataccel = apply_hysteresis(lataccel, last_lataccel, diff)
  last_lataccel = alpha * lataccel + (1 - alpha) * last_lataccel

  output_curvature = last_lataccel / (max(v_ego, 1) ** 2)

  return float(np.interp(v_ego, [5, 10], [apply_curvature, output_curvature]))


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle, lat_active, CP):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = np.clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                              current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, steering_angle, lat_active, CarControllerParams.ANGLE_LIMITS)

  # Ford Q4/CAN FD has more torque available compared to Q3/CAN so we limit it based on lateral acceleration.
  # Safety is not aware of the road roll so we subtract a conservative amount at all times
  if CP.flags & FordFlags.CANFD:
    # Limit curvature to conservative max lateral acceleration
    curvature_accel_limit = MAX_LATERAL_ACCEL / (max(v_ego_raw, 1) ** 2)
    apply_curvature = float(np.clip(apply_curvature, -curvature_accel_limit, curvature_accel_limit))

  return apply_curvature


def apply_creep_compensation(accel: float, v_ego: float) -> float:
  creep_accel = np.interp(v_ego, [1., 3.], [0.6, 0.])
  creep_accel = np.interp(accel, [0., 0.2], [creep_accel, 0.])
  accel -= creep_accel
  return float(accel)


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.CAN = fordcan.CanBus(CP)

    self.params = Params()

    self.apply_curvature_last = 0
    self.anti_overshoot_curvature_last = 0
    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0

    # BluePilot: SubMaster for model data
    self.sm = messaging.SubMaster(['modelV2'])
    self.model = None

    # BluePilot: Predicted curvature blending
    self.curvature_lookup_time = 0.2  # seconds
    self.pc_blend_ratio = 0.20  # 20% predicted, 80% desired

    # BluePilot: Curvature rate computation
    self.curvature_rate_delta_t = 0.3  # seconds
    self.curvature_rate_deque = deque(maxlen=6)  # 0.3s at 20Hz

    # Lane centering via curvature offset (Phase 2, default OFF)
    self.enable_lane_positioning = False
    self.centering_gain = 0.0003  # curvature offset per meter of lane offset

    # BluePilot: Human turn detection and post-reset ramp (Phase 3)
    self.reset_steering_last = False
    self.post_reset_ramp_active = False

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    # BluePilot: Read toggleable params every 1s
    if (self.frame % 100) == 0:
      try:
        self.enable_lane_positioning = self.params.get_bool("enable_lane_positioning")
      except Exception:
        self.enable_lane_positioning = False

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      # BluePilot: Update model data at 20Hz (matches steer step)
      self.sm.update(0)
      if self.sm.updated['modelV2']:
        self.model = self.sm['modelV2']

      apply_curvature_rate = 0.0
      ramp_type = 0  # Slow (inactive default)

      if CC.latActive:
        ramp_type = 2  # Fast (active)

        # Bronco and some other cars consistently overshoot curv requests
        # Apply some deadzone + smoothing convergence to avoid oscillations
        if self.CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_EXPLORER_MK6, CAR.FORD_F_150_MK14):
          self.anti_overshoot_curvature_last = anti_overshoot(actuators.curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
          desired_curvature = self.anti_overshoot_curvature_last
        else:
          desired_curvature = actuators.curvature

        # BluePilot: Predicted curvature blending (40% predicted, 60% desired)
        # Only blend when moving to avoid noise amplification at standstill (orientationRate.z / ~0 = huge)
        if CS.out.vEgoRaw > 1.0 and self.model is not None and len(self.model.orientationRate.z) >= 17:
          curvatures = np.array(self.model.orientationRate.z) / CS.out.vEgoRaw
          predicted_curvature = float(np.interp(self.curvature_lookup_time, ModelConstants.T_IDXS, curvatures))
        else:
          predicted_curvature = desired_curvature

        apply_curvature = (predicted_curvature * self.pc_blend_ratio) + (desired_curvature * (1 - self.pc_blend_ratio))

        # Lane centering: add small curvature offset based on lane position error
        if (self.enable_lane_positioning and self.model is not None
            and len(self.model.laneLines) > 2 and len(self.model.laneLineProbs) > 2
            and CS.out.vEgoRaw > 15.0):
          left_y = self.model.laneLines[1].y[0]
          right_y = self.model.laneLines[2].y[0]
          left_prob = self.model.laneLineProbs[1]
          right_prob = self.model.laneLineProbs[2]
          lane_width = right_y + (-left_y)
          width_tolerance = float(np.interp(lane_width, [3.75, 4.25], [0.81, 0.59]))
          laneline_confidence = min(left_prob, right_prob, width_tolerance)
          if laneline_confidence > 0.6:
            laneline_scale = float(np.interp(laneline_confidence, [0.6, 0.8], [0.0, 1.0]))
            path_offset_lanelines = (left_y + right_y) / 2
            path_offset_position = float(np.interp(0.2, ModelConstants.T_IDXS, self.model.position.y))
            lane_offset = path_offset_position * (1 - laneline_scale) + path_offset_lanelines * laneline_scale
            apply_curvature += lane_offset * self.centering_gain

        # BluePilot: Human turn detection (Phase 3)
        # Only trigger on actual human steering override, NOT at standstill
        human_turn = CS.out.steeringPressed and abs(CS.out.steeringAngleDeg) > 45.0
        reset_steering = human_turn

        if reset_steering:
          apply_curvature = 0.0
          ramp_type = 3  # Immediate
          self.curvature_rate_deque.clear()
          self.post_reset_ramp_active = False
        else:
          # Detect transition out of reset — start post-reset ramp
          if self.reset_steering_last and not reset_steering:
            self.post_reset_ramp_active = True
            self.apply_curvature_last = 0.0

        self.reset_steering_last = reset_steering

        # apply rate limits, curvature error limit, and clip to signal range
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)

        self.apply_curvature_last = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last, current_curvature,
                                                                CS.out.vEgoRaw, 0., CC.latActive, self.CP)

        # Post-reset ramp: gradually ramp curvature from 0, keep path_angle=0 for ford.h bypass
        if self.post_reset_ramp_active:
          self.apply_curvature_last = apply_std_steer_angle_limits(apply_curvature, self.apply_curvature_last,
                                                                   CS.out.vEgoRaw, 0., CC.latActive, CarControllerParams.ANGLE_LIMITS)
          curvature_error = abs(apply_curvature - self.apply_curvature_last)
          curvature_threshold = max(abs(apply_curvature) * 0.1, 0.001)
          if curvature_error < curvature_threshold:
            self.post_reset_ramp_active = False

        if reset_steering:
          self.apply_curvature_last = 0.0

        # BluePilot: Curvature rate from derivative of predicted curvature
        if not reset_steering:
          self.curvature_rate_deque.append(predicted_curvature)
        if CS.out.vEgoRaw > 1.0 and len(self.curvature_rate_deque) > 1:
          delta_t = (self.curvature_rate_delta_t if len(self.curvature_rate_deque) == self.curvature_rate_deque.maxlen
                     else (len(self.curvature_rate_deque) - 1) * 0.05)
          apply_curvature_rate = (self.curvature_rate_deque[-1] - self.curvature_rate_deque[0]) / delta_t / CS.out.vEgoRaw
        else:
          apply_curvature_rate = 0.0

        # Speed gating: disable curvature rate above 15.5 m/s
        speed_factor = float(np.interp(CS.out.vEgoRaw, [0.0, 14.5, 15.5], [1.0, 1.0, 0.0]))
        # Curvature gating: only active for curves > 0.008 1/m
        curv_factor = float(np.interp(abs(predicted_curvature), [0.0, 0.008, 0.01], [0.0, 0.0, 1.0]))
        apply_curvature_rate *= speed_factor * curv_factor
        apply_curvature_rate = float(np.clip(apply_curvature_rate, -0.001023, 0.001023))

        # BluePilot: Lane change handling
        lane_change = self.model is not None and self.model.meta.laneChangeState in (
          LaneChangeState.preLaneChange, LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)
        if lane_change:
          factor = float(np.interp(CS.out.vEgoRaw, [4.4, 40.23], [0.95, 0.85]))
          if self.model.meta.laneChangeDirection == LaneChangeDirection.left and self.apply_curvature_last < 0:
            self.apply_curvature_last *= factor
          elif self.model.meta.laneChangeDirection == LaneChangeDirection.right and self.apply_curvature_last > 0:
            self.apply_curvature_last *= factor
          apply_curvature_rate = 0.0

      else:
        # Not latActive — zero everything
        apply_curvature = actuators.curvature
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        self.apply_curvature_last = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last, current_curvature,
                                                                CS.out.vEgoRaw, 0., CC.latActive, self.CP)
        self.curvature_rate_deque.clear()
        self.reset_steering_last = False
        self.post_reset_ramp_active = False
        apply_curvature_rate = 0.0
        ramp_type = 0

      # Send CAN message: path_offset=0, path_angle=0, curvature+rate NEGATED
      if self.CP.flags & FordFlags.CANFD:
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode,
                         0., 0., -self.apply_curvature_last, -apply_curvature_rate, counter,
                         ramp_type=ramp_type, precision_type=1))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive,
                         0., 0., -self.apply_curvature_last, -apply_curvature_rate,
                         ramp_type=ramp_type, precision_type=1))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      accel = actuators.accel
      gas = accel

      if CC.longActive:
        # Compensate for engine creep at low speed.
        # Either the ABS does not account for engine creep, or the correction is very slow
        # TODO: verify this applies to EV/hybrid
        accel = apply_creep_compensation(accel, CS.out.vEgo)

        # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
        # however even 3.5 m/s^3 causes some overshoot with a step response.
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      gas = float(np.clip(gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      # Both gas and accel are in m/s^2, accel is used solely for braking
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      # PCM applies pitch compensation to gas/accel, but we need to compensate for the brake/pre-charge bits
      accel_due_to_pitch = 0.0
      if len(CC.orientationNED) == 3:
        accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

      accel_pitch_compensated = accel + accel_due_to_pitch
      if accel_pitch_compensated > 0.3 or not CC.longActive:
        self.brake_request = False
      elif accel_pitch_compensated < 0.0:
        self.brake_request = True

      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      # TODO: look into using the actuators packet to send the desired speed
      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping, self.brake_request, v_ego_kph=V_CRUISE_MAX))

      self.accel = accel
      self.gas = gas

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

    # send acc ui msg at 5Hz or if ui state changes
    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      send_ui = True
      self.distance_bar_frame = self.frame

    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      show_distance_bars = self.frame - self.distance_bar_frame < 400
      can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                                 fcw_alert, CS.out.cruiseState.standstill, show_distance_bars,
                                                 hud_control, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars

    new_actuators = actuators.as_builder()
    new_actuators.curvature = self.apply_curvature_last
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
    return new_actuators, can_sends
