import math
import numpy as np
from collections import deque
from cereal import log, messaging
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.carlog import carlog
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


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle, lat_active, CP,
                                curvature_error=None, angle_limits=None):
  if curvature_error is None:
    curvature_error = CarControllerParams.CURVATURE_ERROR
  if angle_limits is None:
    angle_limits = CarControllerParams.ANGLE_LIMITS

  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = np.clip(apply_curvature, current_curvature - curvature_error,
                              current_curvature + curvature_error)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, steering_angle, lat_active, angle_limits)

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
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0

    # BluePilot longitudinal state
    self._bp_long_active_last = False
    self.bp_gas_last = 0.0
    self.bp_accel_last = 0.0
    self.gas_ema = 0.0  # EMA-smoothed gas output to reduce drivetrain jitter
    self.bpSpeedAllow = False
    self.op_brake_actuate_last = False
    self.MAX_URBAN_SPEED_MPH = 45.0
    self.following_accel_ROC = 0.004  # max accel change per scan when in following mode (was 0.002 — too abrupt)
    self.brake_actuate_target = -0.25   # was -0.14 — delayed for Explorer ST aggressive brake pads (2-3x over-delivery)
    self.brake_actuate_release = -0.08  # was -0.06 — slightly wider hysteresis to reduce on/off cycling
    self.precharge_actuate_target = -0.20  # was -0.12 — engage later so pads don't grip too early
    self.precharge_actuate_release = -0.08  # was -0.06
    self.disable_BP_long_UI = False
    self.disable_downhill_comp_UI = True  # disable downhill pitch comp — enabling caused 23% more gas/brake transitions

    # BluePilot: SubMaster for model data and radar
    self.sm = messaging.SubMaster(['modelV2', 'radarState'])
    self.model = None

    # BluePilot: Predicted curvature blending
    self.pc_blend_ratio = 0.30  # 30% predicted, 70% desired (reduced from 0.40 to tame EPAS overshoot)

    # Low-speed curvature stabilizer state: deadband (<4 m/s) + EMA (>4 m/s)
    self.smooth_curvature_last = 0.0

    # BluePilot: Curvature rate computation
    self.curvature_rate_delta_t = 0.3  # seconds
    self.curvature_rate_deque = deque(maxlen=7)  # 0.3s at 20Hz: 7 samples = 6 intervals × 0.05s = 0.30s

    # Lane centering via PI controller (default ON)
    self.enable_lane_positioning = True
    self.lane_centering_integral = 0.0  # PI integral accumulator (m·s)
    self.lane_offset_ema = 0.0  # EMA-smoothed lane offset to filter lane line noise
    self.lc_kp = 0.0001  # curvature per meter of lane offset (P term) — very low, I-term does the work
    self.lc_ki = 0.0002  # curvature per meter·second of accumulated offset (I term) — handles persistent drift smoothly

    # BluePilot: Human turn detection and post-reset ramp (Phase 3)
    self.reset_steering_last = False
    self.post_reset_ramp_active = False

    # FordCurveMode: runtime-switchable curve aggressiveness (0=current, 1=moderate, 2=aggressive)
    self.curve_mode = 0
    self._apply_curve_mode(0)

  def _apply_curve_mode(self, mode):
    """Apply curve mode preset values."""
    preset = CarControllerParams.CURVE_MODE_PARAMS.get(mode, CarControllerParams.CURVE_MODE_PARAMS[0])
    self._curvature_lookup_time_param = preset['curvature_lookup_time']
    self.curvature_rate_gain = preset['curvature_rate_gain']
    self._active_angle_limits = preset['angle_limits']
    self._active_curvature_error = preset['curvature_error']
    self._smooth_tau = preset.get('smooth_tau', (0.12, 0.04))

  def _get_curvature_lookup_time(self, v_ego):
    """Get curvature lookup time, optionally speed-dependent."""
    p = self._curvature_lookup_time_param
    if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], (list, tuple)):
      # Speed-dependent: ([speed_bps], [time_bps])
      return float(np.interp(v_ego, p[0], p[1]))
    return float(p)  # scalar

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    # BluePilot: Read toggleable params every 1s
    if (self.frame % 100) == 0:
      try:
        self.enable_lane_positioning = self.params.get_bool("enable_lane_positioning")
      except Exception:
        self.enable_lane_positioning = False
      try:
        self.disable_BP_long_UI = self.params.get_bool("disable_BP_long_UI")
      except Exception:
        self.disable_BP_long_UI = False
      try:
        self.disable_downhill_comp_UI = self.params.get_bool("disable_downhill_comp_UI")
      except Exception:
        self.disable_downhill_comp_UI = True  # disable downhill pitch comp — enabling caused 23% more gas/brake transitions
      try:
        mode_val = self.params.get("FordCurveMode")
        new_mode = int(mode_val) if mode_val is not None else 0
        new_mode = max(0, min(2, new_mode))
      except Exception:
        new_mode = 0
      if new_mode != self.curve_mode:
        self.curve_mode = new_mode
        self._apply_curve_mode(new_mode)

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
    # Pre-initialize apply_curv_send so it is defined on non-steer frames (frame % STEER_STEP != 0).
    # The steer block runs at 20Hz; on the other 4/5 frames new_actuators.curvature still needs a value.
    apply_curv_send = self.apply_curvature_last
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
        if self.CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_F_150_MK14):
          self.anti_overshoot_curvature_last = anti_overshoot(actuators.curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
          desired_curvature = self.anti_overshoot_curvature_last
        else:
          desired_curvature = actuators.curvature

        # BluePilot: Predicted curvature blending (30% predicted, 70% desired)
        # Only blend when moving to avoid noise amplification at standstill (orientationRate.z / ~0 = huge)
        if CS.out.vEgoRaw > 1.0 and self.model is not None and len(self.model.orientationRate.z) >= len(ModelConstants.T_IDXS):
          curvatures = np.array(self.model.orientationRate.z) / CS.out.vEgoRaw
          lookup_time = self._get_curvature_lookup_time(CS.out.vEgoRaw)
          predicted_curvature = float(np.interp(lookup_time, ModelConstants.T_IDXS, curvatures))
        else:
          predicted_curvature = desired_curvature

        # Speed-dependent blend: reduce predicted curvature influence at low/mid speed
        # to reduce planner-sourced hunting (0.3 Hz oscillation worst at 10-13 m/s)
        blend = float(np.interp(CS.out.vEgoRaw, [7., 22.], [0.10, self.pc_blend_ratio]))
        apply_curvature = (predicted_curvature * blend) + (desired_curvature * (1 - blend))

        # Low-speed curvature stabilizer: deadband + EMA
        #   <4 m/s:   deadband only — ignore corrections smaller than threshold,
        #             passes large/sustained corrections immediately with no lag
        #   4-7 m/s:  shrinking deadband + light EMA transition zone
        #   >7 m/s:   EMA only (mode preset tau), no deadband
        smooth_dt = DT_CTRL * CarControllerParams.STEER_STEP  # 0.05s at 20Hz

        # Speed-dependent deadband threshold (1/m): 0.001 at standstill, 0.0 at 7 m/s
        # Planner noise at 1-3 mph is ~0.013-0.022 amplitude; 0.001 blocks small oscillations
        # while passing gentle curves (0.001-0.002) and real lane corrections
        deadband = float(np.interp(CS.out.vEgoRaw, [0., 4., 7.], [0.001, 0.001, 0.0]))
        if abs(apply_curvature - self.smooth_curvature_last) <= deadband:
          apply_curvature = self.smooth_curvature_last  # hold: change too small to act on
        # EMA above 4 m/s (no EMA below — deadband already holds)
        if CS.out.vEgoRaw >= 4.0:
          smooth_tau = float(np.interp(CS.out.vEgoRaw, [4., 7., 25.], [self._smooth_tau[0], self._smooth_tau[0], self._smooth_tau[1]]))
          smooth_alpha = 1.0 - np.exp(-smooth_dt / smooth_tau)
          apply_curvature = float(smooth_alpha * apply_curvature + (1.0 - smooth_alpha) * self.smooth_curvature_last)
        self.smooth_curvature_last = apply_curvature

        # Lane centering: PI controller on lateral position error
        # P term reacts immediately to current offset; I term cancels persistent structural bias.
        # Integral gates: good lane confidence, low curvature (not in a turn where lane line
        # geometry shifts create false offset), and driver not overriding.
        lc_integral_step = 0.0  # tracks integral increment this frame for anti-windup
        if (self.enable_lane_positioning and self.model is not None
            and len(self.model.laneLines) > 2 and len(self.model.laneLineProbs) > 2
            and CS.out.vEgoRaw > 7.0):
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
            # SIGN VERIFICATION NEEDED: lane_width formula (right_y + (-left_y)) requires right_y > left_y,
            # implying lane line y uses positive-right convention. model.position.y uses OpenPilot's
            # standard positive-left convention. These may have opposite signs for the same offset.
            # Verify from drive logs: when car is visually left of center, are lane_offset > 0
            # and apply_curvature correction positive (= right turn)? If not, negate lane_offset.
            lane_offset_raw = path_offset_position * (1 - laneline_scale) + path_offset_lanelines * laneline_scale
            # EMA-smooth the lane offset to filter lane line detector noise (~0.08m frame-to-frame jitter).
            # tau=1.5s removes high-frequency noise while preserving real drift signals.
            lc_ema_alpha = 1.0 - np.exp(-smooth_dt / 1.5)
            self.lane_offset_ema = float(lc_ema_alpha * lane_offset_raw + (1.0 - lc_ema_alpha) * self.lane_offset_ema)
            lane_offset = self.lane_offset_ema
            # Integral gate: only accumulate on straights when driver is not overriding.
            # Curve gate prevents false integral buildup from lane line geometry shift during turns.
            # P+I CORRECTION is applied unconditionally (whenever lane confidence is good) —
            # the P-term is needed in gentle curves to maintain centering, and the I-term
            # represents structural bias (road crown) that persists through curves.
            if abs(apply_curvature) < 0.005 and not CS.out.steeringPressed:
              lc_integral_step = lane_offset * smooth_dt
              self.lane_centering_integral += lc_integral_step
              self.lane_centering_integral = float(np.clip(self.lane_centering_integral, -0.3, 0.3))
            else:
              self.lane_centering_integral *= 0.98  # decay during curves or driver overrides
            pi_p = self.lc_kp * lane_offset
            pi_i = self.lc_ki * self.lane_centering_integral
            apply_curvature += pi_p + pi_i
            # 1Hz debug log: PI controller internals for drive analysis
            if self.frame % 100 == 0:
              carlog.info("LC: off=%.3f ll=%.3f pos=%.4f scl=%.2f conf=%.2f wid=%.2f int=%.4f P=%.6f I=%.6f curv=%.6f spd=%.1f" % (
                lane_offset, path_offset_lanelines, path_offset_position, laneline_scale, laneline_confidence,
                lane_width, self.lane_centering_integral, pi_p, pi_i, apply_curvature, CS.out.vEgoRaw))
          else:
            self.lane_centering_integral *= 0.98  # decay when lane lines not confident enough
        else:
          self.lane_centering_integral *= 0.98  # decay below speed gate or no model

        # Measured curvature: used for driver override tracking and rate limiting
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)

        # BluePilot: Driver override handling (Ford-specific)
        # Ford EPAS tracks commanded curvature directly. If we keep commanding our desired
        # curvature while the driver steers, the servo actively fights their input.
        # Fix: when steeringPressed, command measured curvature so EPAS is cooperative
        # (holds actual state) instead of fighting toward the model's target.
        #
        # Hard reset (>45°): immediate snap to measured, clear all ramp state
        # Gentle override (<45°): blend toward measured, EMA re-engages smoothly on release
        human_turn = CS.out.steeringPressed and abs(CS.out.steeringAngleDeg) > 45.0
        reset_steering = human_turn

        if reset_steering:
          # Hard reset: snap to measured curvature, not zero (zero causes jerk mid-curve)
          apply_curvature = current_curvature
          self.smooth_curvature_last = current_curvature
          ramp_type = 3  # Immediate
          self.curvature_rate_deque.clear()
          self.post_reset_ramp_active = False
          self.lane_centering_integral = 0.0  # clear integral to prevent snap on re-engage
          self.lane_offset_ema = 0.0  # reset EMA to prevent stale correction
        elif CS.out.steeringPressed:
          # Gentle override: blend toward measured so EPAS stops fighting driver.
          # alpha=0.6 at 20Hz (0.05s/step) ≈ 2-3 cycles to fully track measured.
          # smooth_curvature_last updated here so EMA re-engages from actual position.
          override_alpha = 0.6
          apply_curvature = (override_alpha * current_curvature +
                             (1.0 - override_alpha) * self.smooth_curvature_last)
          self.smooth_curvature_last = apply_curvature
          self.curvature_rate_deque.clear()
        else:
          # Normal: detect transition out of hard reset — start post-reset ramp
          if self.reset_steering_last and not reset_steering:
            self.post_reset_ramp_active = True
            self.apply_curvature_last = current_curvature  # ramp from actual, not zero

        self.reset_steering_last = reset_steering

        # apply rate limits, curvature error limit, and clip to signal range
        apply_curvature_pre_rl = apply_curvature  # save before rate limiting for anti-windup
        self.apply_curvature_last = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last, current_curvature,
                                                                CS.out.vEgoRaw, 0., CC.latActive, self.CP,
                                                                curvature_error=self._active_curvature_error,
                                                                angle_limits=self._active_angle_limits)

        # Anti-windup: if the rate limiter clipped in the same direction the integral was pushing,
        # undo this frame's integral step so the integral doesn't accumulate beyond what the EPAS received.
        if lc_integral_step != 0.0:
          rl_clip = apply_curvature_pre_rl - self.apply_curvature_last
          if rl_clip * lc_integral_step > 0:  # clip and integral step in same direction
            self.lane_centering_integral -= lc_integral_step
            self.lane_centering_integral = float(np.clip(self.lane_centering_integral, -1.0, 1.0))

        # Post-reset ramp: gradually ramp curvature from 0, keep path_angle=0 for ford.h bypass
        if self.post_reset_ramp_active:
          self.apply_curvature_last = apply_std_steer_angle_limits(apply_curvature, self.apply_curvature_last,
                                                                   CS.out.vEgoRaw, 0., CC.latActive, self._active_angle_limits)
          curvature_error = abs(apply_curvature - self.apply_curvature_last)
          curvature_threshold = max(abs(apply_curvature) * 0.1, 0.001)
          if curvature_error < curvature_threshold:
            self.post_reset_ramp_active = False

        if reset_steering:
          self.apply_curvature_last = current_curvature

        # BluePilot: Curvature rate from derivative of predicted curvature
        if not reset_steering:
          self.curvature_rate_deque.append(predicted_curvature)
        if CS.out.vEgoRaw > 1.0 and len(self.curvature_rate_deque) > 1:
          delta_t = (self.curvature_rate_delta_t if len(self.curvature_rate_deque) == self.curvature_rate_deque.maxlen
                     else (len(self.curvature_rate_deque) - 1) * 0.05)
          apply_curvature_rate = (self.curvature_rate_deque[-1] - self.curvature_rate_deque[0]) / delta_t / CS.out.vEgoRaw
        else:
          apply_curvature_rate = 0.0

        # Curvature gating: only active for curves > 0.001 1/m (no speed gate)
        curv_factor = float(np.interp(abs(predicted_curvature), [0.0, 0.001, 0.002], [0.0, 0.0, 1.0]))
        apply_curvature_rate *= curv_factor * self.curvature_rate_gain
        apply_curvature_rate = float(np.clip(apply_curvature_rate, -0.001023, 0.001023))

        # apply_curv_send is the value actually sent to EPAS. All post-rate-limit transformations
        # (lane change factor, FF bias) are applied here only — self.apply_curvature_last must
        # remain unmodified after rate limiting to preserve the rate limiter's state baseline.
        apply_curv_send = self.apply_curvature_last

        # BluePilot: Lane change handling — scale apply_curv_send only, not the rate limiter state
        lane_change = self.model is not None and self.model.meta.laneChangeState in (
          LaneChangeState.preLaneChange, LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)
        if lane_change:
          factor = float(np.interp(CS.out.vEgoRaw, [4.4, 40.23], [0.95, 0.85]))
          if self.model.meta.laneChangeDirection == LaneChangeDirection.left and apply_curv_send < 0:
            apply_curv_send *= factor
          elif self.model.meta.laneChangeDirection == LaneChangeDirection.right and apply_curv_send > 0:
            apply_curv_send *= factor
          apply_curvature_rate = 0.0

        # 1Hz debug log: curvature pipeline stages, override state, anti-windup
        if self.frame % 100 == 0:
          rl_clipped = abs(apply_curvature_pre_rl - self.apply_curvature_last) > 1e-6
          aw_fired = lc_integral_step != 0.0 and rl_clipped and (apply_curvature_pre_rl - self.apply_curvature_last) * lc_integral_step > 0
          carlog.info("CP: des=%.6f pred=%.6f ema=%.6f preRL=%.6f RL=%.6f send=%.6f meas=%.6f | ovr=%d rst=%d ramp=%d rlClip=%d aw=%d | ang=%.1f tq=%.2f" % (
            desired_curvature, predicted_curvature, self.smooth_curvature_last,
            apply_curvature_pre_rl, self.apply_curvature_last, apply_curv_send, current_curvature,
            CS.out.steeringPressed, reset_steering, self.post_reset_ramp_active, rl_clipped, aw_fired,
            CS.out.steeringAngleDeg, CS.out.steeringTorque))

        # Feed-forward EPAS bias correction: DISABLED for now.
        # The sign-flip guard prevents correction when |cmd| < ff_bias (~0.000420), which covers
        # nearly the entire centering range. This means FF does nothing for small corrections
        # and actively harms centering. Re-enable after recalibrating with drive data.
        # if not reset_steering:
        #   ff_bias = float(np.interp(abs(apply_curv_send), [0.0, 0.003], [0.000420, 0.000500]))
        #   if abs(apply_curv_send) > ff_bias:  # prevent sign flip near zero
        #     apply_curv_send -= ff_bias * np.sign(apply_curv_send)

      else:
        # Not latActive — zero everything
        apply_curvature = actuators.curvature
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        self.apply_curvature_last = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last, current_curvature,
                                                                CS.out.vEgoRaw, 0., CC.latActive, self.CP,
                                                                curvature_error=self._active_curvature_error,
                                                                angle_limits=self._active_angle_limits)
        self.curvature_rate_deque.clear()
        self.reset_steering_last = False
        self.post_reset_ramp_active = False
        self.smooth_curvature_last = 0.0
        self.lane_centering_integral = 0.0  # clear on full disengage
        self.lane_offset_ema = 0.0  # reset EMA
        apply_curvature_rate = 0.0
        ramp_type = 0
        apply_curv_send = self.apply_curvature_last

      # Send CAN message: path_offset=0, path_angle=0, curvature+rate NEGATED
      if self.CP.flags & FordFlags.CANFD:
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode,
                         0., 0., -apply_curv_send, -apply_curvature_rate, counter,
                         ramp_type=ramp_type, precision_type=1))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive,
                         0., 0., -apply_curv_send, -apply_curvature_rate,
                         ramp_type=ramp_type, precision_type=1))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # openpilot variable names: accel = brake analog (m/s^2), gas = accelerator analog (m/s^2)
    # brake_actuate = press brakes, precharge_actuate = pre-charge brakes for faster response
    # send acc msg at 50Hz
    v_ego_mph = CS.out.vEgo * 2.23694

    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      # First calculate the stock logic's accel, gas, and brake request
      op_accel = actuators.accel
      op_gas = op_accel

      if CC.longActive:
        # Compensate for engine creep at low speed.
        # Either the ABS does not account for engine creep, or the correction is very slow
        op_accel = apply_creep_compensation(op_accel, CS.out.vEgo)

        # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
        # however even 3.5 m/s^3 causes some overshoot with a step response.
        op_accel = max(op_accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

      op_accel = float(np.clip(op_accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      op_gas = float(np.clip(op_gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      # Both gas and accel are in m/s^2, accel is used solely for braking
      if not CC.longActive or op_gas < CarControllerParams.MIN_GAS:
        op_gas = CarControllerParams.INACTIVE_GAS

      # PCM applies pitch compensation to gas/accel, but we need to compensate for the brake/pre-charge bits
      accel_due_to_pitch = 0.0
      if len(CC.orientationNED) == 3:
        accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

      # Downhill compensation toggle: some Ford vehicles handle pitch natively,
      # so adding accel_due_to_pitch when going downhill double-dips and causes harsh braking
      if self.disable_downhill_comp_UI:
        if accel_due_to_pitch < 0:
          accel_due_to_pitch = 0

      accel_pitch_compensated = op_accel + accel_due_to_pitch
      op_brake_actuate = self.op_brake_actuate_last
      if accel_pitch_compensated > self.brake_actuate_release or not CC.longActive:
        op_brake_actuate = False
      elif accel_pitch_compensated < self.brake_actuate_target:
        op_brake_actuate = True

      stopping = CC.actuators.longControlState == LongCtrlState.stopping

      # Speed deadband for BP long: engage above 50 mph, disallow below 45 mph;
      # 45-50 keeps current state to avoid oscillation.
      bpSpeedTooSlow = v_ego_mph < self.MAX_URBAN_SPEED_MPH
      bpSpeedHighEnough = v_ego_mph > self.MAX_URBAN_SPEED_MPH + 5
      if bpSpeedHighEnough:
        self.bpSpeedAllow = True
      if bpSpeedTooSlow:
        self.bpSpeedAllow = False

      # BluePilot longitudinal: lead-aware gas modulation + rate-limited accel/brake
      if not self.disable_BP_long_UI:

        # Lead time (s) and lead state
        v_ego = max(CS.out.vEgo, 0.5)
        lead_time_sec = 999.0  # no lead: treat as far
        lead = None
        v_rel = 0.0
        v_lead = 0.0
        if self.sm.valid.get('radarState', False):
          rs = self.sm['radarState']
          lead = getattr(rs, 'leadOne', None)
          if lead is not None and getattr(lead, 'status', 0) != 1:
            lead = None
          if lead:
            d_rel = float(getattr(lead, 'dRel', 0))
            v_rel = float(getattr(lead, 'vRel', 0))
            v_lead = float(getattr(lead, 'vLead', 0))
            if d_rel > 0:
              lead_time_sec = d_rel / v_ego
        lead_time_sec = float(np.clip(lead_time_sec, 0.0, 999.0))
        v_lead_mph = v_lead * 2.23694

        # Time to collision
        ttc_sec = 120.0
        if self.sm.valid.get('radarState', False):
          rs = self.sm['radarState']
          lead = getattr(rs, 'leadOne', None)
          if lead is not None and getattr(lead, 'status', 0) != 1:
            lead = None
          if lead:
            d_rel = float(getattr(lead, 'dRel', 0))
            v_rel = float(getattr(lead, 'vRel', 0))
            if d_rel > 0 and v_rel < 0:
              ttc_sec = d_rel / (-v_rel)
            else:
              ttc_sec = 60.0
        ttc_sec = float(np.clip(ttc_sec, 0.2, 120.0))

        # Lead classification and gas/accel limits
        gaining = False
        pacing = False
        trailing = False
        max_follow_gas = op_gas
        min_follow_gas = op_gas
        max_follow_accel = op_accel
        min_follow_accel = op_accel
        bp_brake_actuate = False
        bp_precharge_actuate = False

        # Gaining on lead, pacing, or trailing away
        # Deadband ±0.2 m/s — wider than stock ±0.1 to reduce transitions, but
        # tighter than ±0.3 which let the car get too close (P5 gap 0.96s)
        if lead:
          if v_rel < -0.2:
            gaining = True
          elif v_rel > 0.2:
            trailing = True
          else:
            pacing = True

        # Limits when gaining
        if gaining:
          if lead_time_sec < 2.0:  # was 1.5s — start coasting earlier for more buffer
            max_follow_gas = 0.0
            min_follow_gas = 0.0
          else:
            max_follow_gas = op_gas
            min_follow_gas = op_gas
          max_follow_accel = op_accel
          min_follow_accel = op_accel

        # Limits when pacing
        if pacing:
          max_follow_gas = 0.1 + accel_due_to_pitch  # was 0.2→0.1 — gentle pacing, best balance of comfort and following
          min_follow_gas = 0.0
          max_follow_accel = op_accel
          min_follow_accel = op_accel

        # Limits when trailing
        if trailing:
          max_follow_gas = op_gas
          min_follow_gas = op_gas
          max_follow_accel = op_accel
          min_follow_accel = op_accel

        # Limits with no lead — pass through PID output unchanged
        if lead is None:
          max_follow_gas = op_gas
          min_follow_gas = op_gas
          max_follow_accel = op_accel
          min_follow_accel = op_accel

        # Apply BP gas and accel targets
        bp_gas = float(np.clip(op_gas, min_follow_gas, max_follow_gas))
        bp_accel = float(np.clip(op_accel, min_follow_accel, max_follow_accel))

        # Rate limit braking to dampen initial hit, but only when no imminent collision
        if ttc_sec > 10.0 and lead_time_sec > 0.5:  # was 8.0 — rate limit braking for longer before safety override
          # Rate limit both directions: smooth brake onset AND smooth recovery from braking
          bp_accel = float(np.clip(bp_accel,
                                   self.bp_accel_last - self.following_accel_ROC,
                                   self.bp_accel_last + self.following_accel_ROC))

        # Set brake_actuate and precharge_actuate flags with independent thresholds
        if bp_accel < self.brake_actuate_target:
          bp_brake_actuate = True
        if bp_accel > self.brake_actuate_release:
          bp_brake_actuate = False
        if bp_accel < self.precharge_actuate_target:
          bp_precharge_actuate = True
        if bp_accel > self.precharge_actuate_release:
          bp_precharge_actuate = False

        # Determine if we will use BP long: require highway speed, no pedal override,
        # and when following a lead, require lead > 40 mph (don't coast into traffic jam)
        gasPressed = CS.out.gasPressed
        brakePressed = CS.out.brakePressed
        apply_bp_long = (self.bpSpeedAllow and not gasPressed and not brakePressed
                         and (lead is None or v_lead_mph > 40.0))

        if apply_bp_long and CC.longActive:
          accel = bp_accel
          gas = bp_gas
          brake_actuate = bp_brake_actuate
          precharge_actuate = bp_precharge_actuate
        else:
          accel = op_accel
          gas = op_gas
          brake_actuate = op_brake_actuate
          precharge_actuate = op_brake_actuate

        self.bp_gas_last = bp_gas
        self.bp_accel_last = bp_accel
      else:
        # BP long disabled — use stock logic
        accel = op_accel
        gas = op_gas
        brake_actuate = op_brake_actuate
        precharge_actuate = op_brake_actuate

      # EMA-smooth gas output to reduce drivetrain jitter (9x amplification measured)
      # tau=0.3s at 50Hz (ACC_CONTROL_STEP=2): alpha = 1 - exp(-0.02/0.3) ≈ 0.065
      if CC.longActive and gas > CarControllerParams.MIN_GAS:
        gas_alpha = 0.065
        self.gas_ema = gas_alpha * gas + (1 - gas_alpha) * self.gas_ema
        gas = self.gas_ema
      else:
        self.gas_ema = 0.0

      # No brake and gas at the same time
      if brake_actuate:
        gas = CarControllerParams.INACTIVE_GAS
        self.gas_ema = 0.0

      # Clip to ford.h ACCDATA safety limits
      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      if gas != CarControllerParams.INACTIVE_GAS:
        gas = float(np.clip(gas, CarControllerParams.MIN_GAS, CarControllerParams.ACCEL_MAX))

      # Compensate for Explorer ST aggressive brake pads (measured 2-2.5x over-delivery at light braking)
      # Scale only the CAN brake force value — all control logic (thresholds, rate limiting, TTC bypass)
      # operates on the unscaled accel. This preserves brake actuation timing while reducing force.
      # self.accel stores the UNSCALED value for the jerk limit reference on the next frame.
      accel_for_can = accel
      if accel < 0:
        scale = float(np.interp(-accel, [0.0, 1.2, 2.0], [0.55, 0.55, 1.0]))
        accel_for_can = accel * scale

      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel_for_can, stopping,
                                              brake_actuate, precharge_actuate, v_ego_kph=V_CRUISE_MAX))

      self.accel = accel  # unscaled — used for jerk limit reference on next frame
      self.gas = gas
      self._bp_long_active_last = not self.disable_BP_long_UI
      self.op_brake_actuate_last = op_brake_actuate

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
    new_actuators.curvature = float(apply_curv_send)  # cast: capnp rejects numpy.float64
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
    return new_actuators, can_sends
