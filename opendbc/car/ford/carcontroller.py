import math
import cereal.messaging as messaging
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.lateral import ISO_LATERAL_ACCEL, apply_std_steer_angle_limits
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX
from selfdrive.modeld.constants import ModelConstants

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

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

    # Predicted curvature blending (BluePilot)
    self.sm = messaging.SubMaster(['modelV2', 'drivingModelData'])
    self.model = None
    self.curvature_lookup_time = 0.42  # seconds ahead to sample predicted curvature
    # Blend ratio: 40% predicted curvature, 60% desired curvature (same for CAN and CANFD)
    self.pc_blend_bp = [0.0, 0.001]    # curvature breakpoints (1/m)
    self.pc_blend_v = [0.40, 0.40]     # blend ratio at each breakpoint

    # PID lane centering for path_angle signal (BluePilot)
    self.lc_pid_k_p = 0.25            # Proportional gain
    self.lc_pid_k_i = 0.05            # Integral gain
    self.lc_pid_integral = 0.0
    self.lc_pid_prev_error = 0.0
    self.lc_pid_gain = 5.0            # CAN output scaling (BluePilot LC_PID_GAIN_CAN)
    self.path_angle_max = 0.5         # radians, clamp limit
    self.path_offset_max = 2.0        # meters, clamp limit
    self.lane_conf_threshold = 0.6    # minimum lane line probability for PID

    # Curvature rate signal
    self.curvature_rate_dt = 1.0 / 20.0  # 20 Hz update rate

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    # Update model data from SubMaster
    self.sm.update(0)
    if self.sm.updated['modelV2']:
      self.model = self.sm['modelV2']
    dm_data = self.sm['drivingModelData'] if self.sm.updated['drivingModelData'] else None

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
      # Bronco and some other cars consistently overshoot curv requests
      # Apply some deadzone + smoothing convergence to avoid oscillations
      if self.CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_F_150_MK14):
        self.anti_overshoot_curvature_last = anti_overshoot(actuators.curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
        apply_curvature = self.anti_overshoot_curvature_last
      else:
        apply_curvature = actuators.curvature

      # Blend predicted curvature from model with desired curvature (BluePilot)
      # Predicted curvature leads desired, reducing phase lag through curves
      if self.model is not None and len(self.model.orientationRate.z) >= 17:
        curvatures = np.array(self.model.orientationRate.z) / max(CS.out.vEgoRaw, 0.01)
        predicted_curvature = float(np.interp(self.curvature_lookup_time, ModelConstants.T_IDXS, curvatures))
        blend_ratio = float(np.interp(abs(apply_curvature), self.pc_blend_bp, self.pc_blend_v))
        apply_curvature = predicted_curvature * blend_ratio + apply_curvature * (1.0 - blend_ratio)

      # apply rate limits, curvature error limit, and clip to signal range
      current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)

      self.apply_curvature_last = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last, current_curvature,
                                                              CS.out.vEgoRaw, 0., CC.latActive, self.CP)

      # Compute curvature rate (time derivative of curvature) for EPS anticipation
      apply_curvature_rate = (self.apply_curvature_last - getattr(self, '_prev_apply_curvature', 0.0)) / self.curvature_rate_dt
      apply_curvature_rate = float(np.clip(apply_curvature_rate, -0.001024, 0.00102375))
      self._prev_apply_curvature = self.apply_curvature_last

      # PID lane centering: compute path_angle and path_offset from lane position error (BluePilot)
      apply_path_angle = 0.0
      apply_path_offset = 0.0

      if CC.latActive and dm_data is not None:
        left_y = float(getattr(dm_data.laneLineMeta, 'leftY', math.nan))
        right_y = float(getattr(dm_data.laneLineMeta, 'rightY', math.nan))
        left_prob = float(getattr(dm_data.laneLineMeta, 'leftProb', 0.0))
        right_prob = float(getattr(dm_data.laneLineMeta, 'rightProb', 0.0))

        if (np.isfinite(left_y) and np.isfinite(right_y) and
            left_prob >= self.lane_conf_threshold and right_prob >= self.lane_conf_threshold):
          lane_width = right_y - left_y
          if 2.8 < lane_width < 4.2:
            # Lane center error: positive = right of center
            lane_center_error = 0.5 * (left_y + right_y)

            # PID controller for path_angle
            self.lc_pid_integral += lane_center_error * self.curvature_rate_dt
            self.lc_pid_integral = float(np.clip(self.lc_pid_integral, -2.0, 2.0))  # anti-windup

            pid_output = (self.lc_pid_k_p * lane_center_error +
                          self.lc_pid_k_i * self.lc_pid_integral)

            apply_path_angle = pid_output * self.lc_pid_gain
            apply_path_angle = float(np.clip(apply_path_angle, -self.path_angle_max, self.path_angle_max))

            # Path offset directly from lane center error
            apply_path_offset = float(np.clip(lane_center_error, -self.path_offset_max, self.path_offset_max))

            self.lc_pid_prev_error = lane_center_error
          else:
            # Lane width out of range, reset PID
            self.lc_pid_integral = 0.0
        else:
          # Low confidence, ramp down PID
          self.lc_pid_integral *= 0.95  # gradual decay
      else:
        # Not active, reset PID state
        self.lc_pid_integral = 0.0
        apply_curvature_rate = 0.0

      if self.CP.flags & FordFlags.CANFD:
        # Ford 4-signal lateral control (BluePilot)
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode,
                                                      apply_path_offset, apply_path_angle,
                                                      -self.apply_curvature_last, -apply_curvature_rate, counter))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive,
                                                     apply_path_offset, apply_path_angle,
                                                     -self.apply_curvature_last, -apply_curvature_rate))

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
