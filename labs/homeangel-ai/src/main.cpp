// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/runtime/config_utils.h"
#include "utils/tracker_api.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <functional>
#include <numeric>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace sima_examples {
void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

double time_ms() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(now).count()) /
         1000.0;
}
} // namespace sima_examples

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

/// COCO keypoint names in the order the BoxDecode pose payload emits them.
///
/// The order is fixed by the wire format, not by configuration: `decode_pose` returns a
/// `[N, 17, 3]` tensor whose second axis is positional. Insight joins skeleton edges by
/// name, so these strings are part of the published metadata contract.
constexpr std::array<const char*, 17> kCocoKeypointNames = {
    "nose",           "left_eye",   "right_eye",   "left_ear",   "right_ear",   "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
    "right_hip",      "left_knee",  "right_knee",  "left_ankle", "right_ankle"};

/// Skeleton edges drawn between keypoints, expressed as index pairs into
/// `kCocoKeypointNames`. Mirrors the COCO topology Insight renders.
constexpr std::array<std::pair<int, int>, 17> kCocoSkeleton = {{{0, 1},
                                                                {0, 2},
                                                                {1, 3},
                                                                {2, 4},
                                                                {0, 5},
                                                                {0, 6},
                                                                {5, 7},
                                                                {7, 9},
                                                                {6, 8},
                                                                {8, 10},
                                                                {5, 11},
                                                                {6, 12},
                                                                {11, 12},
                                                                {11, 13},
                                                                {13, 15},
                                                                {12, 14},
                                                                {14, 16}}};

/// One keypoint in source-frame pixel space. `visibility` is the decoder's per-joint
/// confidence in [0, 1]; the debug overlay uses it to hide uncertain joints.
struct Keypoint {
  float x = 0.0f;
  float y = 0.0f;
  float visibility = 0.0f;
};

/// One detected person: the bounding box that anchors the pose plus its 17 keypoints.
struct Pose {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  std::array<Keypoint, 17> keypoints{};
};

struct AlertRoute {
  std::string selector;
  std::string chat_id;
};

struct AppConfig {
  std::string device_id = "modalix-hallway-01";
  std::string zone_label = "bedroom";
  std::string model_path;
  std::vector<std::string> rtsp_urls;
  /// Encoded RTSP path used for every stream in this application.
  simaai::neat::nodes::groups::RtspCodec codec = simaai::neat::nodes::groups::RtspCodec::H264;
  int latency_ms = 100;
  bool tcp = true;
  int input_max_width = 1920;
  int input_max_height = 1080;
  int source_width = 1280;
  int source_height = 720;
  int source_fps = 30;
  int frames = 0;
  int max_inflight_per_stream = 4;
  int max_inflight_total = 16;
  double min_score = 0.50;
  double nms_iou = 0.60;
  int max_poses = 50;
  double min_keypoint_visibility = 0.30;
  bool profile = false;
  int warmup_frames = 30;
  std::string insight_host = "127.0.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  bool video_enabled = true;
  fs::path events_log = "/workspace/labs/homeangel-ai/events.log";
  fs::path telemetry_json = "/workspace/labs/homeangel-ai/telemetry.json";
  int telemetry_history_limit = 150;
  std::string insight_viewer_url = "https://192.168.1.10:8081/static/viewer.html?mode=light&src=0&max_channels=4";
  std::string webhook_url;
  bool log_state_transitions = false;
  int log_features_every_n_frames = 0;
  double fall_velocity_threshold = 0.45;
  double angular_velocity_threshold_deg_s = 90.0;
  double angle_threshold_deg = 60.0;
  double confirm_seconds = 2.0;
  double stillness_threshold = 0.04;
  double velocity_window_seconds = 0.75;
  double stillness_window_seconds = 0.8;
  double down_centroid_y_fraction = 0.55;
  double recovery_angle_deg = 35.0;
  int min_present_frames = 3;
  float tracking_iou_threshold = 0.3f;
  int tracking_max_missing_frames = 15;
  bool telegram_enabled = false;
  std::string telegram_token_env = "HOMEANGEL_TELEGRAM_BOT_TOKEN";
  std::vector<AlertRoute> telegram_routes;
  bool vlm_enabled = false;
  std::string vlm_host = "127.0.0.1";
  int vlm_port = 9998;
  std::vector<std::string> vlm_models;
  std::string vlm_model_paths;
  int vlm_max_tokens = 96;
  double vlm_timeout_seconds = 20.0;
  int vlm_jpeg_quality = 78;
  double vlm_crop_padding = 0.20;
  std::string vlm_system_prompt =
      "You are HomeAngel's local safety analyst. Describe only visible safety context. "
      "Do not identify the person.";
  std::string vlm_user_prompt =
      "Briefly describe whether the person appears to be on the floor, motionless, "
      "getting up, sitting normally, or needing help.";
};

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

simaai::neat::nodes::groups::RtspCodec parse_input_codec(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "h264" || lowered == "avc" || lowered == "h.264") {
    return simaai::neat::nodes::groups::RtspCodec::H264;
  }
  if (lowered == "h265" || lowered == "hevc" || lowered == "h.265") {
    return simaai::neat::nodes::groups::RtspCodec::H265;
  }
  throw std::runtime_error("input.codec must be h264/avc or h265/hevc");
}

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

struct ProfileWindow {
  bool enabled = false;
  int stream_index = 0;
  int interval = 100;
  int frames = 0;
  int poses = 0;
  double start_ms = 0.0;
  double detection_pull_ms = 0.0;
  double metadata_send_ms = 0.0;

  void add(double detection_pull, double metadata_send, int pose_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    poses += pose_count;
    detection_pull_ms += detection_pull;
    metadata_send_ms += metadata_send;
    if (frames >= interval)
      flush();
  }

  void flush() {
    if (!enabled || frames == 0)
      return;
    const double elapsed = sima_examples::time_ms() - start_ms;
    const double output_fps = elapsed > 0.0 ? static_cast<double>(frames) * 1000.0 / elapsed : 0.0;
    const auto avg = [this](double value) { return value / static_cast<double>(frames); };
    std::cout << "[profile stream=" << stream_index << "] frames=" << frames
              << " output_fps=" << output_fps << " avg_detection_pull_ms=" << avg(detection_pull_ms)
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_poses=" << static_cast<double>(poses) / static_cast<double>(frames) << "\n";
    frames = 0;
    poses = 0;
    start_ms = 0.0;
    detection_pull_ms = 0.0;
    metadata_send_ms = 0.0;
  }
};

struct PerformanceSummary {
  int total_frames = 0;
  bool vlm_enabled = false;
  double first_ms = 0.0;
  double last_ms = 0.0;
  std::vector<double> latencies_ms;

  void add(double latency_ms) {
    const double now = sima_examples::time_ms();
    if (total_frames == 0) {
      first_ms = now;
    }
    last_ms = now;
    ++total_frames;
    latencies_ms.push_back(latency_ms);
  }

  void print() const {
    const double elapsed_s = last_ms > first_ms ? (last_ms - first_ms) / 1000.0 : 0.0;
    const double fps = elapsed_s > 0.0 ? static_cast<double>(total_frames) / elapsed_s : 0.0;
    double avg_latency = 0.0;
    double p95_latency = 0.0;
    if (!latencies_ms.empty()) {
      avg_latency =
          std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
          static_cast<double>(latencies_ms.size());
      auto sorted = latencies_ms;
      std::sort(sorted.begin(), sorted.end());
      const std::size_t p95_index =
          std::min(sorted.size() - 1, static_cast<std::size_t>(std::ceil(sorted.size() * 0.95)) - 1);
      p95_latency = sorted[p95_index];
    }

    std::cout << "[summary] avg_fps=" << std::fixed << std::setprecision(2) << fps
              << " avg_latency_ms=" << avg_latency << " p95_latency_ms=" << p95_latency
              << " total_frames=" << total_frames
              << " compute=decode:EV74,pose:MLA,tracking:CPU,fall_logic:CPU,alerts:CPU,vlm:"
              << (vlm_enabled ? "LLiMa-local" : "disabled") << "\n";
  }
};

enum class FallState {
  Upright,
  Falling,
  Down,
  FallConfirmed,
};

const char* state_label(FallState state) {
  switch (state) {
  case FallState::Upright:
    return "UPRIGHT";
  case FallState::Falling:
    return "FALLING";
  case FallState::Down:
    return "DOWN";
  case FallState::FallConfirmed:
    return "FALL DETECTED";
  }
  return "UPRIGHT";
}

const char* telemetry_state_label(FallState state) {
  switch (state) {
  case FallState::Upright:
    return "UPRIGHT";
  case FallState::Falling:
    return "FALLING";
  case FallState::Down:
    return "DOWN";
  case FallState::FallConfirmed:
    return "FALL_CONFIRMED";
  }
  return "UPRIGHT";
}

int state_rank(FallState state) {
  switch (state) {
  case FallState::Upright:
    return 0;
  case FallState::Falling:
    return 1;
  case FallState::Down:
    return 2;
  case FallState::FallConfirmed:
    return 3;
  }
  return 0;
}

struct PoseFeatures {
  double timestamp_s = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  double bbox_w = 0.0;
  double bbox_h = 1.0;
  double torso_angle_deg = 0.0;
  bool low = false;
};

struct TrackedPose {
  int track_id = 0;
  Pose pose;
  FallState state = FallState::Upright;
  bool confirmed_now = false;
  double confidence = 0.0;
  double vertical_velocity = 0.0;
  double angular_velocity = 0.0;
  double torso_angle_deg = 0.0;
};

struct TrackFallMemory {
  FallState state = FallState::Upright;
  int present_frames = 0;
  int missing_frames = 0;
  bool event_emitted = false;
  double down_since_s = -1.0;
  double still_since_s = -1.0;
  std::deque<PoseFeatures> history;
};

bool visible_keypoint(const Pose& pose, std::size_t index, double min_visibility) {
  return index < pose.keypoints.size() && pose.keypoints[index].visibility >= min_visibility;
}

std::pair<double, double> midpoint(const Keypoint& a, const Keypoint& b) {
  return {(static_cast<double>(a.x) + static_cast<double>(b.x)) * 0.5,
          (static_cast<double>(a.y) + static_cast<double>(b.y)) * 0.5};
}

PoseFeatures compute_pose_features(const Pose& pose, double timestamp_s, int frame_h,
                                   const AppConfig& cfg) {
  PoseFeatures features;
  features.timestamp_s = timestamp_s;
  features.bbox_w = std::max(1.0, static_cast<double>(pose.x2 - pose.x1));
  features.bbox_h = std::max(1.0, static_cast<double>(pose.y2 - pose.y1));
  features.cx = (static_cast<double>(pose.x1) + static_cast<double>(pose.x2)) * 0.5;
  features.cy = (static_cast<double>(pose.y1) + static_cast<double>(pose.y2)) * 0.5;

  const bool left_hip = visible_keypoint(pose, 11, cfg.min_keypoint_visibility);
  const bool right_hip = visible_keypoint(pose, 12, cfg.min_keypoint_visibility);
  if (left_hip && right_hip) {
    const auto hip = midpoint(pose.keypoints[11], pose.keypoints[12]);
    features.cx = hip.first;
    features.cy = hip.second;
  } else if (left_hip) {
    features.cx = pose.keypoints[11].x;
    features.cy = pose.keypoints[11].y;
  } else if (right_hip) {
    features.cx = pose.keypoints[12].x;
    features.cy = pose.keypoints[12].y;
  }

  const bool left_shoulder = visible_keypoint(pose, 5, cfg.min_keypoint_visibility);
  const bool right_shoulder = visible_keypoint(pose, 6, cfg.min_keypoint_visibility);
  if ((left_hip || right_hip) && (left_shoulder || right_shoulder)) {
    std::pair<double, double> hip = {features.cx, features.cy};
    if (left_hip && right_hip) {
      hip = midpoint(pose.keypoints[11], pose.keypoints[12]);
    }
    std::pair<double, double> shoulder = hip;
    if (left_shoulder && right_shoulder) {
      shoulder = midpoint(pose.keypoints[5], pose.keypoints[6]);
    } else if (left_shoulder) {
      shoulder = {pose.keypoints[5].x, pose.keypoints[5].y};
    } else {
      shoulder = {pose.keypoints[6].x, pose.keypoints[6].y};
    }
    const double dx = shoulder.first - hip.first;
    const double dy = shoulder.second - hip.second;
    const double mag = std::max(1.0, std::hypot(dx, dy));
    const double cos_to_vertical = std::clamp((-dy) / mag, -1.0, 1.0);
    features.torso_angle_deg = std::acos(cos_to_vertical) * 180.0 / M_PI;
  } else {
    features.torso_angle_deg =
        std::atan2(features.bbox_w, features.bbox_h) * 180.0 / M_PI;
  }
  features.low =
      frame_h > 0 && (features.cy / static_cast<double>(frame_h)) >= cfg.down_centroid_y_fraction;
  return features;
}

const PoseFeatures& window_reference(const std::deque<PoseFeatures>& history,
                                     double current_s, double window_s) {
  const PoseFeatures* ref = &history.front();
  for (const auto& item : history) {
    if (current_s - item.timestamp_s <= window_s) {
      return item;
    }
    ref = &item;
  }
  return *ref;
}

double normalized_motion(const std::deque<PoseFeatures>& history, const PoseFeatures& current,
                         double window_s) {
  double max_motion = 0.0;
  for (const auto& item : history) {
    if (current.timestamp_s - item.timestamp_s > window_s) {
      continue;
    }
    const double dist = std::hypot(current.cx - item.cx, current.cy - item.cy);
    const double scale = std::max(1.0, (current.bbox_h + item.bbox_h) * 0.5);
    max_motion = std::max(max_motion, dist / scale);
  }
  return max_motion;
}

double confidence_for_confirmation(const AppConfig& cfg, double vertical_velocity,
                                   double angular_velocity, double angle_deg,
                                   double still_held_s) {
  const double v_score = std::clamp(vertical_velocity / cfg.fall_velocity_threshold, 0.0, 1.0);
  const double a_score =
      std::clamp(angular_velocity / cfg.angular_velocity_threshold_deg_s, 0.0, 1.0);
  const double angle_score = std::clamp((angle_deg - 45.0) / 45.0, 0.0, 1.0);
  const double still_score =
      cfg.confirm_seconds <= 0.0 ? 1.0 : std::clamp(still_held_s / cfg.confirm_seconds, 0.0, 1.0);
  return std::clamp(0.35 * std::max(v_score, a_score) + 0.30 * angle_score +
                        0.35 * still_score,
                    0.0, 1.0);
}

class FallDetector {
public:
  TrackedPose update(int track_id, const Pose& pose, double timestamp_s, int frame_h,
                     const AppConfig& cfg) {
    auto& memory = tracks_[track_id];
    memory.present_frames += 1;
    memory.missing_frames = 0;

    PoseFeatures current = compute_pose_features(pose, timestamp_s, frame_h, cfg);
    memory.history.push_back(current);
    while (!memory.history.empty() &&
           current.timestamp_s - memory.history.front().timestamp_s >
               std::max(cfg.velocity_window_seconds, cfg.stillness_window_seconds) + 0.5) {
      memory.history.pop_front();
    }

    const PoseFeatures& ref =
        window_reference(memory.history, current.timestamp_s, cfg.velocity_window_seconds);
    const double dt = std::max(1e-3, current.timestamp_s - ref.timestamp_s);
    const double bbox_h = std::max(1.0, (current.bbox_h + ref.bbox_h) * 0.5);
    const double vertical_velocity = (current.cy - ref.cy) / bbox_h / dt;
    const double angular_velocity = std::max(0.0, current.torso_angle_deg - ref.torso_angle_deg) / dt;
    const double still_motion =
        normalized_motion(memory.history, current, cfg.stillness_window_seconds);
    const bool still = still_motion <= cfg.stillness_threshold;
    const bool tilted = current.torso_angle_deg >= cfg.angle_threshold_deg;
    const bool wide = current.bbox_w > current.bbox_h * 1.15;
    const bool low = current.low;
    const bool recovered = current.torso_angle_deg <= cfg.recovery_angle_deg && !low && !wide;
    const bool fast_motion = vertical_velocity >= cfg.fall_velocity_threshold ||
                             angular_velocity >= cfg.angular_velocity_threshold_deg_s;

    bool confirmed_now = false;
    double still_held_s = memory.still_since_s >= 0.0 ? current.timestamp_s - memory.still_since_s
                                                      : 0.0;
    const FallState previous_state = memory.state;

    if (memory.present_frames < cfg.min_present_frames) {
      memory.state = FallState::Upright;
    } else if (recovered) {
      memory.state = FallState::Upright;
      memory.down_since_s = -1.0;
      memory.still_since_s = -1.0;
      memory.event_emitted = false;
    } else {
      switch (memory.state) {
      case FallState::Upright:
        if (fast_motion && (current.torso_angle_deg >= 45.0 || wide || low)) {
          memory.state = FallState::Falling;
        }
        break;
      case FallState::Falling:
        if ((tilted || wide) && low) {
          memory.state = FallState::Down;
          memory.down_since_s = current.timestamp_s;
          memory.still_since_s = still ? current.timestamp_s : -1.0;
        } else if (!fast_motion && current.torso_angle_deg < 45.0 && !low) {
          memory.state = FallState::Upright;
        }
        break;
      case FallState::Down:
        if (still) {
          if (memory.still_since_s < 0.0) {
            memory.still_since_s = current.timestamp_s;
          }
        } else {
          memory.still_since_s = -1.0;
        }
        still_held_s = memory.still_since_s >= 0.0 ? current.timestamp_s - memory.still_since_s
                                                   : 0.0;
        if (!memory.event_emitted && still_held_s >= cfg.confirm_seconds) {
          memory.state = FallState::FallConfirmed;
          memory.event_emitted = true;
          confirmed_now = true;
        }
        break;
      case FallState::FallConfirmed:
        break;
      }

      if (memory.state == FallState::Falling && (tilted || wide) && low) {
        memory.state = FallState::Down;
        memory.down_since_s = current.timestamp_s;
        memory.still_since_s = still ? current.timestamp_s : -1.0;
      }
    }

    if (cfg.log_state_transitions && previous_state != memory.state) {
      std::cout << std::fixed << std::setprecision(3)
                << "[state track=" << track_id << "] t=" << current.timestamp_s << " "
                << state_label(previous_state) << "->" << state_label(memory.state)
                << " v=" << vertical_velocity << " angle_v=" << angular_velocity
                << " angle=" << current.torso_angle_deg << " cy="
                << (frame_h > 0 ? current.cy / static_cast<double>(frame_h) : 0.0)
                << " still_motion=" << still_motion << " still=" << (still ? "true" : "false")
                << " low=" << (low ? "true" : "false") << " wide=" << (wide ? "true" : "false")
                << "\n";
    }
    if (cfg.log_features_every_n_frames > 0 &&
        memory.present_frames % cfg.log_features_every_n_frames == 0) {
      std::cout << std::fixed << std::setprecision(3)
                << "[features track=" << track_id << "] t=" << current.timestamp_s
                << " state=" << state_label(memory.state) << " v=" << vertical_velocity
                << " angle_v=" << angular_velocity << " angle=" << current.torso_angle_deg
                << " cy=" << (frame_h > 0 ? current.cy / static_cast<double>(frame_h) : 0.0)
                << " still_motion=" << still_motion << " still=" << (still ? "true" : "false")
                << " low=" << (low ? "true" : "false") << " wide=" << (wide ? "true" : "false")
                << " posescore=" << pose.score << "\n";
    }

    TrackedPose tracked;
    tracked.track_id = track_id;
    tracked.pose = pose;
    tracked.state = memory.state;
    tracked.confirmed_now = confirmed_now;
    tracked.vertical_velocity = vertical_velocity;
    tracked.angular_velocity = angular_velocity;
    tracked.torso_angle_deg = current.torso_angle_deg;
    tracked.confidence = confidence_for_confirmation(cfg, vertical_velocity, angular_velocity,
                                                     current.torso_angle_deg, still_held_s);
    return tracked;
  }

  void retain_only(const std::vector<int>& active_track_ids, int max_missing_frames) {
    std::unordered_map<int, bool> active;
    for (int id : active_track_ids) {
      active[id] = true;
    }
    for (auto it = tracks_.begin(); it != tracks_.end();) {
      if (active.find(it->first) == active.end()) {
        it->second.missing_frames += 1;
        if (it->second.missing_frames > max_missing_frames) {
          it = tracks_.erase(it);
        } else {
          ++it;
        }
      } else {
        ++it;
      }
    }
  }

private:
  std::unordered_map<int, TrackFallMemory> tracks_;
};

int run_process(const std::vector<std::string>& args) {
  if (args.empty()) {
    return -1;
  }
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);

  const pid_t pid = fork();
  if (pid == 0) {
    execvp(argv[0], argv.data());
    _exit(127);
  }
  if (pid < 0) {
    return -1;
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    return -1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return -1;
}

struct ProcessResult {
  int exit_code = -1;
  std::string stdout_text;
};

bool write_all_fd(int fd, const std::string& data) {
  const char* ptr = data.data();
  std::size_t remaining = data.size();
  while (remaining > 0) {
    const ssize_t written = write(fd, ptr, remaining);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return errno == EPIPE;
    }
    if (written == 0) {
      return false;
    }
    ptr += written;
    remaining -= static_cast<std::size_t>(written);
  }
  return true;
}

ProcessResult run_process_capture(const std::vector<std::string>& args,
                                  const std::string& stdin_text) {
  ProcessResult result;
  if (args.empty()) {
    return result;
  }

  int stdin_pipe[2] = {-1, -1};
  int stdout_pipe[2] = {-1, -1};
  if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
    if (stdin_pipe[0] >= 0) {
      close(stdin_pipe[0]);
    }
    if (stdin_pipe[1] >= 0) {
      close(stdin_pipe[1]);
    }
    if (stdout_pipe[0] >= 0) {
      close(stdout_pipe[0]);
    }
    if (stdout_pipe[1] >= 0) {
      close(stdout_pipe[1]);
    }
    return result;
  }

  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);

  const pid_t pid = fork();
  if (pid == 0) {
    dup2(stdin_pipe[0], STDIN_FILENO);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    close(stdin_pipe[0]);
    close(stdin_pipe[1]);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    execvp(argv[0], argv.data());
    _exit(127);
  }
  if (pid < 0) {
    close(stdin_pipe[0]);
    close(stdin_pipe[1]);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    return result;
  }

  close(stdin_pipe[0]);
  close(stdout_pipe[1]);
  (void)write_all_fd(stdin_pipe[1], stdin_text);
  close(stdin_pipe[1]);

  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t n = read(stdout_pipe[0], buffer.data(), buffer.size());
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    if (n == 0) {
      break;
    }
    result.stdout_text.append(buffer.data(), static_cast<std::size_t>(n));
  }
  close(stdout_pipe[0]);

  int status = 0;
  if (waitpid(pid, &status, 0) >= 0 && WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  }
  return result;
}

const simaai::neat::Sample* find_field(const simaai::neat::Sample& sample,
                                       const std::string& label) {
  if (sample.stream_label == label) {
    return &sample;
  }
  for (const auto& field : sample.fields) {
    if (const auto* found = find_field(field, label)) {
      return found;
    }
  }
  return nullptr;
}

const simaai::neat::Sample& joined_field(const simaai::neat::Sample& sample,
                                         const std::string& label, std::size_t bundle_index) {
  if (const auto* field = find_field(sample, label)) {
    return *field;
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle && sample.fields.size() > bundle_index) {
    return sample.fields[bundle_index];
  }
  throw std::runtime_error("joined output missing " + label + " field");
}

const simaai::neat::Sample& pose_sample_from_output(const AppConfig& cfg,
                                                    const simaai::neat::Sample& sample) {
  return cfg.vlm_enabled ? joined_field(sample, "poses", 1U) : sample;
}

std::string base64_encode(const std::vector<unsigned char>& data) {
  static constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((data.size() + 2) / 3) * 4);
  for (std::size_t i = 0; i < data.size(); i += 3) {
    const unsigned int b0 = data[i];
    const unsigned int b1 = i + 1 < data.size() ? data[i + 1] : 0;
    const unsigned int b2 = i + 2 < data.size() ? data[i + 2] : 0;
    const unsigned int triple = (b0 << 16) | (b1 << 8) | b2;
    out.push_back(kAlphabet[(triple >> 18) & 0x3F]);
    out.push_back(kAlphabet[(triple >> 12) & 0x3F]);
    out.push_back(i + 1 < data.size() ? kAlphabet[(triple >> 6) & 0x3F] : '=');
    out.push_back(i + 2 < data.size() ? kAlphabet[triple & 0x3F] : '=');
  }
  return out;
}

std::string compact_text(std::string text, std::size_t max_chars = 260) {
  std::string compact;
  compact.reserve(text.size());
  bool previous_space = false;
  for (char c : text) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isspace(uc)) {
      if (!previous_space) {
        compact.push_back(' ');
        previous_space = true;
      }
    } else {
      compact.push_back(c);
      previous_space = false;
    }
  }
  compact = sima_examples::trim_copy(compact);
  if (compact.size() > max_chars) {
    compact.resize(max_chars);
    compact = sima_examples::trim_copy(compact) + "...";
  }
  return compact;
}

std::optional<std::string> extract_text_content(const nlohmann::json& content) {
  if (content.is_string()) {
    return content.get<std::string>();
  }
  if (content.is_array()) {
    std::string joined;
    for (const auto& item : content) {
      if (item.is_string()) {
        joined += item.get<std::string>();
      } else if (item.is_object() && item.contains("text") && item["text"].is_string()) {
        joined += item["text"].get<std::string>();
      }
      if (!joined.empty() && joined.back() != ' ') {
        joined.push_back(' ');
      }
    }
    joined = compact_text(joined);
    if (!joined.empty()) {
      return joined;
    }
  }
  return std::nullopt;
}

std::optional<std::string> extract_completion_text_from_json(const nlohmann::json& doc) {
  if (!doc.contains("choices") || !doc["choices"].is_array() || doc["choices"].empty()) {
    return std::nullopt;
  }
  const auto& choice = doc["choices"].front();
  if (choice.contains("message") && choice["message"].is_object() &&
      choice["message"].contains("content")) {
    if (auto text = extract_text_content(choice["message"]["content"])) {
      return compact_text(*text);
    }
  }
  if (choice.contains("delta") && choice["delta"].is_object() && choice["delta"].contains("content")) {
    if (auto text = extract_text_content(choice["delta"]["content"])) {
      return compact_text(*text);
    }
  }
  if (choice.contains("text") && choice["text"].is_string()) {
    return compact_text(choice["text"].get<std::string>());
  }
  return std::nullopt;
}

std::optional<std::string> extract_completion_text(const std::string& body) {
  const std::string trimmed = sima_examples::trim_copy(body);
  if (trimmed.empty()) {
    return std::nullopt;
  }
  if (trimmed.rfind("data:", 0) == 0 || trimmed.find("\ndata:") != std::string::npos) {
    std::stringstream input(trimmed);
    std::string line;
    std::string merged;
    while (std::getline(input, line)) {
      line = sima_examples::trim_copy(line);
      if (line.rfind("data:", 0) != 0) {
        continue;
      }
      std::string payload = sima_examples::trim_copy(line.substr(5));
      if (payload.empty() || payload == "[DONE]") {
        continue;
      }
      try {
        if (auto text = extract_completion_text_from_json(nlohmann::json::parse(payload))) {
          merged += *text;
          if (!merged.empty() && merged.back() != ' ') {
            merged.push_back(' ');
          }
        }
      } catch (...) {
      }
    }
    merged = compact_text(merged);
    if (!merged.empty()) {
      return merged;
    }
    return std::nullopt;
  }
  try {
    return extract_completion_text_from_json(nlohmann::json::parse(trimmed));
  } catch (const std::exception& ex) {
    std::cerr << "[warn] VLM response was not valid JSON: " << ex.what() << "\n";
    return std::nullopt;
  }
}

class VlmRouter {
public:
  explicit VlmRouter(AppConfig cfg)
      : cfg_(std::move(cfg)),
        endpoint_("http://" + cfg_.vlm_host + ":" + std::to_string(cfg_.vlm_port) +
                  "/v1/chat/completions") {}

  std::optional<std::string> describe_confirmed_fall(const simaai::neat::Sample& output_sample,
                                                     const TrackedPose& tracked) const {
    if (!cfg_.vlm_enabled) {
      return std::nullopt;
    }
    try {
      const auto tensors =
          simaai::neat::tensors_from_sample(joined_field(output_sample, "frame", 0U), true);
      if (tensors.empty()) {
        return std::nullopt;
      }
      cv::Mat frame_bgr =
          tensors.front().to_cv_mat_copy(simaai::neat::ImageSpec::PixelFormat::BGR);
      cv::Mat crop = crop_person(frame_bgr, tracked.pose);
      if (crop.empty()) {
        return std::nullopt;
      }
      return describe_crop(crop);
    } catch (const std::exception& ex) {
      std::cerr << "[warn] VLM confirmation skipped: " << ex.what() << "\n";
      return std::nullopt;
    }
  }

private:
  cv::Mat crop_person(const cv::Mat& frame_bgr, const Pose& pose) const {
    if (frame_bgr.empty()) {
      return {};
    }
    const double width = std::max(1.0, static_cast<double>(pose.x2 - pose.x1));
    const double height = std::max(1.0, static_cast<double>(pose.y2 - pose.y1));
    const double pad_x = width * cfg_.vlm_crop_padding;
    const double pad_y = height * cfg_.vlm_crop_padding;
    const int x1 = std::clamp(static_cast<int>(std::floor(pose.x1 - pad_x)), 0,
                              std::max(0, frame_bgr.cols - 1));
    const int y1 = std::clamp(static_cast<int>(std::floor(pose.y1 - pad_y)), 0,
                              std::max(0, frame_bgr.rows - 1));
    const int x2 = std::clamp(static_cast<int>(std::ceil(pose.x2 + pad_x)), x1 + 1,
                              frame_bgr.cols);
    const int y2 = std::clamp(static_cast<int>(std::ceil(pose.y2 + pad_y)), y1 + 1,
                              frame_bgr.rows);
    return frame_bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
  }

  std::optional<std::string> describe_crop(const cv::Mat& crop_bgr) const {
    std::vector<unsigned char> jpeg;
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, cfg_.vlm_jpeg_quality};
    if (!cv::imencode(".jpg", crop_bgr, jpeg, params)) {
      std::cerr << "[warn] failed to encode VLM crop in memory\n";
      return std::nullopt;
    }
    const std::string data_url = "data:image/jpeg;base64," + base64_encode(jpeg);

    for (const auto& model : cfg_.vlm_models) {
      if (auto description = describe_with_model(model, data_url)) {
        std::cout << "[vlm] description generated by " << model << "\n";
        return description;
      }
    }
    std::cerr << "[warn] no configured local VLM produced a description\n";
    return std::nullopt;
  }

  std::optional<std::string> describe_with_model(const std::string& model,
                                                 const std::string& data_url) const {
    nlohmann::json payload = {
        {"model", model},
        {"stream", false},
        {"max_tokens", cfg_.vlm_max_tokens},
        {"messages",
         nlohmann::json::array(
             {{{"role", "system"}, {"content", cfg_.vlm_system_prompt}},
              {{"role", "user"},
               {"content",
                nlohmann::json::array({{{"type", "image_url"},
                                         {"image_url", {{"url", data_url}}}},
                                        {{"type", "text"}, {"text", cfg_.vlm_user_prompt}}})}}})}};

    std::ostringstream timeout;
    timeout << std::fixed << std::setprecision(1) << cfg_.vlm_timeout_seconds;
    const ProcessResult result =
        run_process_capture({"curl",
                             "-fsS",
                             "-m",
                             timeout.str(),
                             "-H",
                             "Content-Type: application/json",
                             "--data-binary",
                             "@-",
                             endpoint_},
                            payload.dump());
    if (result.exit_code != 0) {
      std::cerr << "[warn] VLM model " << model << " request failed with exit code "
                << result.exit_code << "\n";
      return std::nullopt;
    }
    auto text = extract_completion_text(result.stdout_text);
    if (!text || text->empty()) {
      std::cerr << "[warn] VLM model " << model << " returned no usable text\n";
      return std::nullopt;
    }
    return text;
  }

  AppConfig cfg_;
  std::string endpoint_;
};

class AlertSink {
public:
  explicit AlertSink(AppConfig cfg) : cfg_(std::move(cfg)) {}

  void emit_fall(double timestamp_s, int track_id, double confidence,
                 const std::optional<std::string>& description = std::nullopt) {
    nlohmann::json event = {
        {"event", "fall_detected"},
        {"device_id", cfg_.device_id},
        {"zone_label", cfg_.zone_label},
        {"timestamp_s", std::round(timestamp_s * 1000.0) / 1000.0},
        {"track_id", track_id},
        {"confidence", std::round(std::clamp(confidence, 0.0, 1.0) * 1000.0) / 1000.0},
    };
    if (description && !description->empty()) {
      event["description"] = *description;
    }

    fs::create_directories(cfg_.events_log.parent_path());
    std::ofstream out(cfg_.events_log, std::ios::app);
    if (!out.is_open()) {
      std::cerr << "[warn] failed to open event log: " << cfg_.events_log << "\n";
      return;
    }
    out << event.dump() << "\n";
    out.close();

    if (!cfg_.webhook_url.empty()) {
      post_json(cfg_.webhook_url, event.dump(), "webhook");
    }
    if (cfg_.telegram_enabled) {
      send_telegram(event);
    }
  }

private:
  void post_json(const std::string& url, const std::string& json, const char* label) const {
    const int rc = run_process({"curl", "-fsS", "-m", "5", "-H", "Content-Type: application/json",
                                "-d", json, url});
    if (rc != 0) {
      std::cerr << "[warn] " << label << " POST failed with exit code " << rc << "\n";
    }
  }

  std::vector<std::string> telegram_chat_ids() const {
    std::vector<std::string> chat_ids;
    for (const auto& route : cfg_.telegram_routes) {
      if (route.selector == "*" || route.selector == cfg_.zone_label ||
          route.selector == cfg_.device_id) {
        chat_ids.push_back(route.chat_id);
      }
    }
    return chat_ids;
  }

  void send_telegram(const nlohmann::json& event) const {
    const char* token = std::getenv(cfg_.telegram_token_env.c_str());
    if (token == nullptr || std::strlen(token) == 0) {
      std::cerr << "[warn] Telegram enabled but " << cfg_.telegram_token_env << " is not set\n";
      return;
    }
    const auto chat_ids = telegram_chat_ids();
    if (chat_ids.empty()) {
      return;
    }

    const std::string url = std::string("https://api.telegram.org/bot") + token + "/sendMessage";
    const std::string text = "HomeAngel AI fall detected in " + cfg_.zone_label +
                             " (device " + cfg_.device_id + ", track " +
                             std::to_string(event.at("track_id").get<int>()) + ", confidence " +
                             std::to_string(event.at("confidence").get<double>()) + ")" +
                             (event.contains("description")
                                  ? "\nScene: " + event.at("description").get<std::string>()
                                  : "");

    for (const auto& chat_id : chat_ids) {
      nlohmann::json body = {{"chat_id", chat_id}, {"text", text}};
      post_json(url, body.dump(), "Telegram sendMessage");
    }
  }

  AppConfig cfg_;
};

struct StreamRuntime {
  int index = 0;
  simaai::neat::nodes::groups::RtspDecodedInputOptions source_options;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  ProfileWindow profile;
  multi_stream_people_tracker::PeopleTracker tracker;
  FallDetector fall_detector;
  int frame_w = 0;
  int frame_h = 0;
  int fps = 0;
  int processed = 0;
};

double rounded(double value, double scale = 1000.0) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::round(value * scale) / scale;
}

struct TelemetryHistorySample {
  double timestamp_s = 0.0;
  int track_id = 0;
  FallState state = FallState::Upright;
  double vertical_velocity = 0.0;
  double angular_velocity = 0.0;
  double torso_angle_deg = 0.0;
  double confidence = 0.0;
};

class TelemetryPublisher {
public:
  explicit TelemetryPublisher(const AppConfig& cfg)
      : path_(cfg.telemetry_json), history_limit_(std::max(1, cfg.telemetry_history_limit)) {}

  void publish(const AppConfig& cfg, const StreamRuntime& stream, double timestamp_s,
               double latency_ms, double metadata_send_ms,
               const std::vector<TrackedPose>& tracked_poses) {
    if (path_.empty()) {
      return;
    }

    const double now_ms = sima_examples::time_ms();
    frame_wall_times_ms_.push_back(now_ms);
    while (!frame_wall_times_ms_.empty() && now_ms - frame_wall_times_ms_.front() > 1000.0) {
      frame_wall_times_ms_.pop_front();
    }
    const double live_fps =
        frame_wall_times_ms_.size() > 1 && frame_wall_times_ms_.back() > frame_wall_times_ms_.front()
            ? (static_cast<double>(frame_wall_times_ms_.size() - 1) * 1000.0) /
                  (frame_wall_times_ms_.back() - frame_wall_times_ms_.front())
            : 0.0;

    FallState global_state = FallState::Upright;
    for (const auto& tracked : tracked_poses) {
      if (state_rank(tracked.state) > state_rank(global_state)) {
        global_state = tracked.state;
      }
      history_.push_back(TelemetryHistorySample{timestamp_s,
                                                tracked.track_id,
                                                tracked.state,
                                                tracked.vertical_velocity,
                                                tracked.angular_velocity,
                                                tracked.torso_angle_deg,
                                                tracked.confidence});
    }
    while (history_.size() > static_cast<std::size_t>(history_limit_)) {
      history_.pop_front();
    }

    nlohmann::json people = nlohmann::json::array();
    for (const auto& tracked : tracked_poses) {
      const auto& pose = tracked.pose;
      people.push_back({{"track_id", tracked.track_id},
                        {"state", telemetry_state_label(tracked.state)},
                        {"display_state", state_label(tracked.state)},
                        {"vertical_velocity", rounded(tracked.vertical_velocity)},
                        {"angular_velocity_deg_s", rounded(tracked.angular_velocity)},
                        {"torso_angle_deg", rounded(tracked.torso_angle_deg)},
                        {"confidence", rounded(std::clamp(tracked.confidence, 0.0, 1.0))},
                        {"bbox",
                         {std::lround(pose.x1), std::lround(pose.y1),
                          std::lround(std::max(0.0f, pose.x2 - pose.x1)),
                          std::lround(std::max(0.0f, pose.y2 - pose.y1))}}});
    }

    nlohmann::json history = nlohmann::json::array();
    for (const auto& item : history_) {
      history.push_back({{"timestamp_s", rounded(item.timestamp_s)},
                         {"track_id", item.track_id},
                         {"state", telemetry_state_label(item.state)},
                         {"vertical_velocity", rounded(item.vertical_velocity)},
                         {"angular_velocity_deg_s", rounded(item.angular_velocity)},
                         {"torso_angle_deg", rounded(item.torso_angle_deg)},
                         {"confidence", rounded(std::clamp(item.confidence, 0.0, 1.0))}});
    }

    nlohmann::json doc = {
        {"schema", "homeangel.telemetry.v1"},
        {"updated_at_s", rounded(timestamp_s)},
        {"device_id", cfg.device_id},
        {"zone_label", cfg.zone_label},
        {"stream_index", stream.index},
        {"frame", {{"processed", stream.processed}, {"width", stream.frame_w}, {"height", stream.frame_h}}},
        {"state", telemetry_state_label(global_state)},
        {"people", std::move(people)},
        {"history", std::move(history)},
        {"thresholds",
         {{"fall_velocity_threshold", rounded(cfg.fall_velocity_threshold)},
          {"angular_velocity_threshold_deg_s", rounded(cfg.angular_velocity_threshold_deg_s)},
          {"angle_threshold_deg", rounded(cfg.angle_threshold_deg)},
          {"confirm_seconds", rounded(cfg.confirm_seconds)},
          {"stillness_threshold", rounded(cfg.stillness_threshold)}}},
        {"performance",
         {{"fps", rounded(live_fps, 100.0)},
          {"latency_ms", rounded(latency_ms, 100.0)},
          {"metadata_send_ms", rounded(metadata_send_ms, 100.0)},
          {"compute",
           {{"decode", "EV74"},
            {"pose", "MLA"},
            {"tracking", "CPU"},
            {"fall_logic", "CPU"},
            {"alerts", "CPU"},
            {"vlm", cfg.vlm_enabled ? "LLiMa GenAI server (loopback)" : "disabled"}}},
          {"power", {{"available", false}, {"label", "not reported"}}}}},
        {"vlm",
         {{"enabled", cfg.vlm_enabled},
          {"host", cfg.vlm_host},
          {"port", cfg.vlm_port},
          {"models", cfg.vlm_models}}},
        {"video",
         {{"mode", "insight_iframe"},
          {"local_only", true},
          {"frames_included", false},
          {"viewer_url", cfg.insight_viewer_url}}},
        {"privacy",
         {{"telemetry_local_only", true},
          {"event_stream_json_only", true},
          {"notifier_frame_access", false},
          {"raw_frames_in_telemetry", false}}}};

    write_atomic(doc);
  }

private:
  void write_atomic(const nlohmann::json& doc) {
    try {
      const auto parent = path_.parent_path();
      if (!parent.empty()) {
        fs::create_directories(parent);
      }
      fs::path tmp_path = path_;
      tmp_path += ".tmp";
      {
        std::ofstream out(tmp_path, std::ios::trunc);
        if (!out.is_open()) {
          throw std::runtime_error("open failed");
        }
        out << std::setw(2) << doc << "\n";
      }
      fs::rename(tmp_path, path_);
    } catch (const std::exception& ex) {
      if (!warned_) {
        std::cerr << "[warn] failed to write telemetry JSON " << path_ << ": " << ex.what()
                  << "\n";
        warned_ = true;
      }
    }
  }

  fs::path path_;
  int history_limit_ = 150;
  bool warned_ = false;
  std::deque<TelemetryHistorySample> history_;
  std::deque<double> frame_wall_times_ms_;
};

struct AppRuntime {
  simaai::neat::Graph graph;
  simaai::neat::Run run;
  std::unique_ptr<simaai::neat::Model> model;
  std::vector<StreamRuntime> streams;
  PerformanceSummary performance;
  std::unique_ptr<AlertSink> alert_sink;
  std::unique_ptr<TelemetryPublisher> telemetry;
  std::unique_ptr<VlmRouter> vlm_router;
  std::string output_name = "poses";
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  options.config_path = "/workspace/labs/homeangel-ai/config.yaml";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[++i];
    } else if (arg == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>] [--validate-config-only]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

std::string strip_inline_comment(const std::string& line) {
  bool in_single = false;
  bool in_double = false;
  std::string out;
  out.reserve(line.size());
  for (char c : line) {
    if (c == '\'' && !in_double) {
      in_single = !in_single;
    } else if (c == '"' && !in_single) {
      in_double = !in_double;
    } else if (c == '#' && !in_single && !in_double) {
      break;
    }
    out.push_back(c);
  }
  return out;
}

std::string unquote(std::string value) {
  value = sima_examples::trim_copy(value);
  if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                            (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::vector<std::string> parse_streams(const fs::path& config_path) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file: " + config_path.string());
  }

  std::vector<std::string> streams;
  bool in_streams = false;
  int streams_indent = -1;
  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const std::string line_without_comment = strip_inline_comment(raw_line);
    if (sima_examples::trim_copy(line_without_comment).empty()) {
      continue;
    }

    int indent = 0;
    while (indent < static_cast<int>(line_without_comment.size()) &&
           (line_without_comment[static_cast<std::size_t>(indent)] == ' ' ||
            line_without_comment[static_cast<std::size_t>(indent)] == '\t')) {
      ++indent;
    }
    const std::string line = sima_examples::trim_copy(line_without_comment);

    if (in_streams && indent <= streams_indent && line.rfind("- ", 0) != 0) {
      in_streams = false;
    }
    if (!in_streams && line == "streams:") {
      in_streams = true;
      streams_indent = indent;
      continue;
    }
    if (in_streams && line.rfind("- ", 0) == 0) {
      const std::string value = unquote(line.substr(2));
      if (value.empty()) {
        throw std::runtime_error("streams entries must be non-empty strings");
      }
      streams.push_back(value);
    }
  }
  if (streams.empty()) {
    throw std::runtime_error("streams must be a non-empty list");
  }
  return streams;
}

std::vector<std::string> split_csv(const std::string& value) {
  std::vector<std::string> out;
  std::stringstream input(value);
  std::string item;
  while (std::getline(input, item, ',')) {
    item = sima_examples::trim_copy(item);
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

bool is_loopback_host(const std::string& host) {
  const std::string lowered = lower_copy(sima_examples::trim_copy(host));
  return lowered == "127.0.0.1" || lowered == "localhost" || lowered == "::1" ||
         lowered == "[::1]";
}

std::vector<AlertRoute> parse_alert_routes(const std::string& value) {
  std::vector<AlertRoute> routes;
  for (const auto& item : split_csv(value)) {
    const std::size_t sep = item.find('=');
    if (sep == std::string::npos) {
      throw std::runtime_error("alerts.telegram.routes entries must use selector=chat_id");
    }
    AlertRoute route;
    route.selector = sima_examples::trim_copy(item.substr(0, sep));
    route.chat_id = sima_examples::trim_copy(item.substr(sep + 1));
    if (route.selector.empty() || route.chat_id.empty()) {
      throw std::runtime_error("alerts.telegram.routes entries must have selector and chat_id");
    }
    routes.push_back(std::move(route));
  }
  return routes;
}

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.device_id.empty(), "device_id must be set");
  sima_examples::require(!cfg.zone_label.empty(), "zone_label must be set");
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  sima_examples::require(!cfg.rtsp_urls.empty(), "streams must be set");
  sima_examples::require(cfg.rtsp_urls.size() <= 4, "this phase supports up to four streams");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(!cfg.events_log.empty(), "output.events_log must be set");
  sima_examples::require(!cfg.telemetry_json.empty(), "output.telemetry_json must be set");
  sima_examples::require(cfg.telemetry_history_limit > 0,
                         "output.telemetry_history_limit must be > 0");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.input_max_width > 0, "input.max_width must be > 0");
  sima_examples::require(cfg.input_max_height > 0, "input.max_height must be > 0");
  sima_examples::require(cfg.source_width > 0, "input.source_width must be > 0");
  sima_examples::require(cfg.source_height > 0, "input.source_height must be > 0");
  sima_examples::require(cfg.source_fps > 0, "input.source_fps must be > 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.max_inflight_per_stream == -1 || cfg.max_inflight_per_stream > 0,
                         "inference.max_inflight_per_stream must be -1 or > 0");
  sima_examples::require(cfg.max_inflight_total == -1 || cfg.max_inflight_total > 0,
                         "inference.max_inflight_total must be -1 or > 0");
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_poses > 0, "inference.max_poses must be > 0");
  sima_examples::require(cfg.min_keypoint_visibility >= 0.0 && cfg.min_keypoint_visibility <= 1.0,
                         "output.min_keypoint_visibility must be between 0 and 1");
  sima_examples::require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  sima_examples::require(cfg.video_port_base > 0, "output.insight.video_port_base must be > 0");
  sima_examples::require(cfg.metadata_port_base > 0,
                         "output.insight.metadata_port_base must be > 0");
  sima_examples::require(cfg.log_features_every_n_frames >= 0,
                         "output.log_features_every_n_frames must be >= 0");
  sima_examples::require(cfg.fall_velocity_threshold > 0.0,
                         "fall_velocity_threshold must be > 0");
  sima_examples::require(cfg.angular_velocity_threshold_deg_s > 0.0,
                         "angular_velocity_threshold_deg_s must be > 0");
  sima_examples::require(cfg.angle_threshold_deg > 0.0 && cfg.angle_threshold_deg < 180.0,
                         "angle_threshold_deg must be between 0 and 180");
  sima_examples::require(cfg.confirm_seconds >= 0.0, "confirm_seconds must be >= 0");
  sima_examples::require(cfg.stillness_threshold > 0.0,
                         "stillness_threshold must be > 0");
  sima_examples::require(cfg.velocity_window_seconds > 0.0,
                         "velocity_window_seconds must be > 0");
  sima_examples::require(cfg.stillness_window_seconds > 0.0,
                         "stillness_window_seconds must be > 0");
  sima_examples::require(cfg.down_centroid_y_fraction >= 0.0 &&
                             cfg.down_centroid_y_fraction <= 1.0,
                         "down_centroid_y_fraction must be between 0 and 1");
  sima_examples::require(cfg.recovery_angle_deg >= 0.0 && cfg.recovery_angle_deg < 180.0,
                         "recovery_angle_deg must be between 0 and 180");
  sima_examples::require(cfg.min_present_frames >= 1, "min_present_frames must be >= 1");
  sima_examples::require(cfg.tracking_iou_threshold >= 0.0f && cfg.tracking_iou_threshold <= 1.0f,
                         "tracking.iou_threshold must be between 0 and 1");
  sima_examples::require(cfg.tracking_max_missing_frames >= 0,
                         "tracking.max_missing_frames must be >= 0");
  if (cfg.telegram_enabled) {
    sima_examples::require(!cfg.telegram_token_env.empty(),
                           "alerts.telegram.token_env must be set when Telegram is enabled");
    sima_examples::require(!cfg.telegram_routes.empty(),
                           "alerts.telegram.routes must be set when Telegram is enabled");
  }
  if (cfg.vlm_enabled) {
    sima_examples::require(cfg.rtsp_urls.size() == 1,
                           "vlm.enabled currently supports a single stream for frame/pose joins");
    sima_examples::require(is_loopback_host(cfg.vlm_host),
                           "vlm.host must be 127.0.0.1/localhost so frame crops stay local");
    sima_examples::require(cfg.vlm_port > 0 && cfg.vlm_port <= 65535,
                           "vlm.port must be between 1 and 65535");
    sima_examples::require(!cfg.vlm_models.empty(),
                           "vlm.models must list at least one local GenAI model when enabled");
    sima_examples::require(cfg.vlm_max_tokens > 0, "vlm.max_tokens must be > 0");
    sima_examples::require(cfg.vlm_timeout_seconds > 0.0,
                           "vlm.timeout_seconds must be > 0");
    sima_examples::require(cfg.vlm_jpeg_quality >= 1 && cfg.vlm_jpeg_quality <= 100,
                           "vlm.jpeg_quality must be between 1 and 100");
    sima_examples::require(cfg.vlm_crop_padding >= 0.0,
                           "vlm.crop_padding must be >= 0");
    sima_examples::require(!cfg.vlm_system_prompt.empty(),
                           "vlm.system_prompt must be set when VLM is enabled");
    sima_examples::require(!cfg.vlm_user_prompt.empty(),
                           "vlm.user_prompt must be set when VLM is enabled");
  }
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);

  AppConfig cfg;
  cfg.device_id = raw.string_or("device_id", "modalix-hallway-01");
  cfg.zone_label = raw.string_or("zone_label", "bedroom");
  cfg.model_path = raw.string_or("model.path", "");
  cfg.rtsp_urls = parse_streams(config_path);
  cfg.codec = parse_input_codec(raw.string_or("input.codec", "h264"));
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.input_max_width = raw.int_or("input.max_width", 1920);
  cfg.input_max_height = raw.int_or("input.max_height", 1080);
  cfg.source_width = raw.int_or("input.source_width", 1280);
  cfg.source_height = raw.int_or("input.source_height", 720);
  cfg.source_fps = raw.int_or("input.source_fps", 30);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.max_inflight_per_stream = raw.int_or("inference.max_inflight_per_stream", 4);
  cfg.max_inflight_total = raw.int_or("inference.max_inflight_total", 16);
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_poses = raw.int_or("inference.max_poses", 50);
  cfg.min_keypoint_visibility = raw.double_or("output.min_keypoint_visibility", 0.30);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  cfg.events_log = raw.string_or("output.events_log", "/workspace/labs/homeangel-ai/events.log");
  cfg.telemetry_json =
      raw.string_or("output.telemetry_json", "/workspace/labs/homeangel-ai/telemetry.json");
  cfg.telemetry_history_limit = raw.int_or("output.telemetry_history_limit", 150);
  cfg.insight_viewer_url = raw.string_or(
      "output.insight.viewer_url",
      "https://192.168.1.10:8081/static/viewer.html?mode=light&src=0&max_channels=4");
  cfg.webhook_url = raw.string_or("webhook_url", "");
  cfg.log_state_transitions = raw.bool_or("output.log_state_transitions", false);
  cfg.log_features_every_n_frames = raw.int_or("output.log_features_every_n_frames", 0);
  cfg.fall_velocity_threshold = raw.double_or("fall_velocity_threshold", 0.45);
  cfg.angular_velocity_threshold_deg_s = raw.double_or("angular_velocity_threshold_deg_s", 90.0);
  cfg.angle_threshold_deg = raw.double_or("angle_threshold_deg", 60.0);
  cfg.confirm_seconds = raw.double_or("confirm_seconds", 2.0);
  cfg.stillness_threshold = raw.double_or("stillness_threshold", 0.04);
  cfg.velocity_window_seconds = raw.double_or("velocity_window_seconds", 0.75);
  cfg.stillness_window_seconds = raw.double_or("stillness_window_seconds", 0.8);
  cfg.down_centroid_y_fraction = raw.double_or("down_centroid_y_fraction", 0.55);
  cfg.recovery_angle_deg = raw.double_or("recovery_angle_deg", 35.0);
  cfg.min_present_frames = raw.int_or("min_present_frames", 3);
  cfg.tracking_iou_threshold = static_cast<float>(raw.double_or("tracking.iou_threshold", 0.3));
  cfg.tracking_max_missing_frames = raw.int_or("tracking.max_missing_frames", 15);
  cfg.telegram_enabled = raw.bool_or("alerts.telegram.enabled", false);
  cfg.telegram_token_env =
      raw.string_or("alerts.telegram.token_env", "HOMEANGEL_TELEGRAM_BOT_TOKEN");
  cfg.telegram_routes = parse_alert_routes(raw.string_or("alerts.telegram.routes", ""));
  cfg.vlm_enabled = raw.bool_or("vlm.enabled", false);
  cfg.vlm_host = raw.string_or("vlm.host", "127.0.0.1");
  cfg.vlm_port = raw.int_or("vlm.port", 9998);
  cfg.vlm_models =
      split_csv(raw.string_or("vlm.models", "Gemma-4-E4B-it,Qwen3-VL-4B-Instruct-GPTQ-a16w4"));
  cfg.vlm_model_paths = raw.string_or("vlm.model_paths", "");
  cfg.vlm_max_tokens = raw.int_or("vlm.max_tokens", 96);
  cfg.vlm_timeout_seconds = raw.double_or("vlm.timeout_seconds", 20.0);
  cfg.vlm_jpeg_quality = raw.int_or("vlm.jpeg_quality", 78);
  cfg.vlm_crop_padding = raw.double_or("vlm.crop_padding", 0.20);
  cfg.vlm_system_prompt =
      raw.string_or("vlm.system_prompt", cfg.vlm_system_prompt);
  cfg.vlm_user_prompt = raw.string_or("vlm.user_prompt", cfg.vlm_user_prompt);
  validate_config(cfg);
  return cfg;
}

std::vector<float> tensor_floats(const simaai::neat::Tensor& tensor) {
  // A frame with no person decodes to a zero-row tensor. Copying one throws, because a
  // zero-byte payload has nothing to map, so treat "no rows" as an empty result.
  const int64_t elements = std::accumulate(tensor.shape.begin(), tensor.shape.end(), int64_t{1},
                                           std::multiplies<int64_t>());
  if (tensor.shape.empty() || elements == 0) {
    return {};
  }

  const auto bytes = tensor.contiguous().copy_payload_bytes();
  std::vector<float> values(bytes.size() / sizeof(float));
  if (!values.empty()) {
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
  }
  return values;
}

/// Decode one pose sample into `Pose` records in source-frame pixel space.
///
/// `decode_pose` returns boxes as `[N, 6]` (x1, y1, x2, y2, score, class_id) and keypoints
/// as `[N, 17, 3]` (x, y, visibility), positionally aligned. Passing the frame size clamps
/// box coordinates to the frame; keypoints are emitted by the decoder unclamped, so the
/// visibility floor and the drawing code are what keep stray joints off the overlay.
std::vector<Pose> decode_poses(const simaai::neat::Sample& sample, int frame_w, int frame_h,
                               int max_poses) {
  const auto tensors = simaai::neat::tensors_from_sample(sample, false);
  if (tensors.empty()) {
    throw std::runtime_error("pose sample carried no tensors");
  }

  const auto decoded = simaai::neat::decode_pose(tensors, frame_w, frame_h, max_poses, false);
  std::vector<Pose> poses;
  for (const auto& item : decoded) {
    const auto boxes = tensor_floats(item.boxes);
    const auto keypoints = tensor_floats(item.keypoints);
    if (boxes.size() % 6U != 0 || keypoints.size() % (17U * 3U) != 0) {
      throw std::runtime_error("pose decode returned malformed tensor sizes");
    }
    const std::size_t box_count = boxes.size() / 6U;
    const std::size_t keypoint_count = keypoints.size() / (17U * 3U);
    if (box_count != keypoint_count) {
      throw std::runtime_error("pose decode returned " + std::to_string(box_count) + " boxes but " +
                               std::to_string(keypoint_count) + " keypoint sets");
    }

    for (std::size_t i = 0; i < box_count && poses.size() < static_cast<std::size_t>(max_poses);
         ++i) {
      const float* box = boxes.data() + i * 6U;
      Pose pose;
      pose.x1 = box[0];
      pose.y1 = box[1];
      pose.x2 = box[2];
      pose.y2 = box[3];
      pose.score = box[4];
      if (std::lround(box[5]) != 0) {
        throw std::runtime_error("pose decode returned a non-person class id");
      }

      const float* points = keypoints.data() + i * 17U * 3U;
      for (std::size_t k = 0; k < 17U; ++k) {
        pose.keypoints[k] = Keypoint{points[k * 3U], points[k * 3U + 1U], points[k * 3U + 2U]};
      }
      poses.push_back(pose);
    }
  }
  return poses;
}

/// Serialize poses into the `data` object Insight's `pose-estimation` overlay consumes.
///
/// Pixel rounding and three-decimal confidence preserve overlay precision while keeping the
/// configured 50-pose maximum within Core's logical metadata-message limit.
std::string overlay_label(const AppConfig& cfg, const TrackedPose& tracked) {
  if (tracked.state == FallState::FallConfirmed) {
    return "FALL DETECTED - " + cfg.zone_label;
  }
  return std::string(state_label(tracked.state));
}

std::string pose_metadata_data_json(const AppConfig& cfg,
                                    const std::vector<TrackedPose>& tracked_poses) {
  nlohmann::json data;
  data["poses"] = nlohmann::json::array();
  for (const auto& tracked : tracked_poses) {
    const auto& pose = tracked.pose;
    nlohmann::json keypoints = nlohmann::json::array();
    for (std::size_t k = 0; k < pose.keypoints.size(); ++k) {
      const Keypoint& point = pose.keypoints[k];
      keypoints.push_back({{"name", kCocoKeypointNames[k]},
                           {"x", std::lround(point.x)},
                           {"y", std::lround(point.y)},
                           {"confidence", std::round(point.visibility * 1000.0f) / 1000.0f}});
    }
    data["poses"].push_back({{"id", "track_" + std::to_string(tracked.track_id)},
                             {"label", overlay_label(cfg, tracked)},
                             {"track_id", tracked.track_id},
                             {"state", state_label(tracked.state)},
                             {"confidence", std::round(pose.score * 1000.0f) / 1000.0f},
                             {"bbox",
                              {std::lround(pose.x1), std::lround(pose.y1),
                               std::lround(std::max(0.0f, pose.x2 - pose.x1)),
                               std::lround(std::max(0.0f, pose.y2 - pose.y1))}},
                             {"keypoints", std::move(keypoints)}});
  }
  return data.dump();
}

std::string tracking_metadata_data_json(const AppConfig& cfg,
                                        const std::vector<TrackedPose>& tracked_poses) {
  nlohmann::json data;
  data["tracks"] = nlohmann::json::array();
  for (const auto& tracked : tracked_poses) {
    const auto& pose = tracked.pose;
    data["tracks"].push_back(
        {{"id", tracked.track_id},
         {"label", overlay_label(cfg, tracked)},
         {"confidence", std::round(std::clamp(tracked.confidence, 0.0, 1.0) * 1000.0) / 1000.0},
         {"bbox",
          {std::lround(pose.x1), std::lround(pose.y1),
           std::lround(std::max(0.0f, pose.x2 - pose.x1)),
           std::lround(std::max(0.0f, pose.y2 - pose.y1))}}});
  }
  return data.dump();
}

simaai::neat::nodes::groups::RtspDecodedInputOptions
build_source_options(const AppConfig& cfg, const std::string& url, int& fps_out, int& width_out,
                     int& height_out) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.auto_caps_from_stream = true;
  opt.codec = cfg.codec;
  opt.dec_width = cfg.source_width;
  opt.dec_height = cfg.source_height;
  if (cfg.codec == simaai::neat::nodes::groups::RtspCodec::H264) {
    opt.fallback_h264_width = cfg.source_width;
    opt.fallback_h264_height = cfg.source_height;
  }
  opt.source_fps = cfg.source_fps;
  width_out = cfg.source_width;
  height_out = cfg.source_height;
  fps_out = cfg.source_fps;
  if (width_out > 0 && height_out > 0 && fps_out > 0) {
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = width_out;
    opt.output_caps.height = height_out;
    opt.output_caps.fps = fps_out;
    opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  }
  return opt;
}

bool output_caps_enabled(
    const simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps& caps) {
  return caps.enable || caps.width > 0 || caps.height > 0 || caps.fps > 0;
}

simaai::neat::FormatTag encoded_format_tag(simaai::neat::nodes::groups::RtspCodec codec) {
  return codec == simaai::neat::nodes::groups::RtspCodec::H265 ? simaai::neat::FormatTag::H265
                                                               : simaai::neat::FormatTag::H264;
}

simaai::neat::InputOptions
encoded_decode_input_options(simaai::neat::nodes::groups::RtspCodec codec) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  opt.format = encoded_format_tag(codec);
  opt.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
  return opt;
}

simaai::neat::InputOptions
encoded_video_input_options(simaai::neat::nodes::groups::RtspCodec codec) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  opt.format = encoded_format_tag(codec);
  opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  return opt;
}

simaai::neat::Graph
build_encoded_source_graph(const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::Graph source("rtsp_encoded_source");

  simaai::neat::nodes::groups::RtspEncodedInputOptions encoded_opt;
  encoded_opt.url = opt.url;
  encoded_opt.codec = opt.codec;
  encoded_opt.latency_ms = opt.latency_ms;
  encoded_opt.tcp = opt.tcp;
  encoded_opt.source_fps = opt.source_fps;
  if (opt.codec == simaai::neat::nodes::groups::RtspCodec::H264) {
    encoded_opt.fallback_h264_width = opt.fallback_h264_width;
    encoded_opt.fallback_h264_height = opt.fallback_h264_height;
  }
  source.add(simaai::neat::nodes::groups::RtspEncodedInput(encoded_opt));
  return source;
}

simaai::neat::Graph
build_decode_graph(const std::string& input_name,
                   const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::Graph decode("decode");
  const bool use_h265 = opt.codec == simaai::neat::nodes::groups::RtspCodec::H265;

  simaai::neat::SimaDecodeOptions dec;
  dec.type = use_h265 ? simaai::neat::SimaDecodeType::H265 : simaai::neat::SimaDecodeType::H264;
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = opt.dec_width;
  dec.dec_height = opt.dec_height;
  dec.dec_fps = opt.source_fps;
  dec.num_buffers = opt.num_buffers;
  dec.input_buffers = opt.decoder_input_buffers;
  dec.decoder_tuning = opt.decoder_tuning;
  dec.memory_opt = opt.decoder_memory_opt;

  decode.connect(simaai::neat::nodes::Input(input_name, encoded_decode_input_options(opt.codec)),
                 simaai::neat::nodes::SimaDecode(dec));
  if (opt.use_videoconvert) {
    decode.add(simaai::neat::nodes::VideoConvert());
  }
  if (opt.use_videoscale) {
    decode.add(simaai::neat::nodes::VideoScale());
  }
  if (output_caps_enabled(opt.output_caps)) {
    const auto& caps = opt.output_caps;
    decode.add(
        simaai::neat::nodes::CapsRaw(caps.format, caps.width, caps.height, caps.fps, caps.memory));
  }
  if (!opt.extra_fragment.empty()) {
    decode.add(simaai::neat::nodes::Custom(opt.extra_fragment));
  }
  return decode;
}

simaai::neat::Graph
build_video_sender_graph(const std::string& input_name,
                         simaai::neat::nodes::groups::RtspCodec codec,
                         const simaai::neat::nodes::groups::VideoSenderOptions& video_options) {
  simaai::neat::Graph video("video_sender");
  video.connect(simaai::neat::nodes::Input(input_name, encoded_video_input_options(codec)),
                simaai::neat::nodes::groups::VideoSender(video_options));
  return video;
}

std::unique_ptr<simaai::neat::Model> build_model(const AppConfig& cfg) {
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_opt.preprocess.input_max_width = cfg.input_max_width;
  model_opt.preprocess.input_max_height = cfg.input_max_height;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV26Pose;
  // YOLO26 pose ships single-class ("person") score heads. The packaged MPK still declares
  // the 80-class detector metadata, so state the real class count rather than inheriting it.
  model_opt.num_classes = 1;
  model_opt.score_threshold = cfg.min_score;
  model_opt.nms_iou_threshold = cfg.nms_iou;
  model_opt.top_k = cfg.max_poses;
  return std::make_unique<simaai::neat::Model>(cfg.model_path, model_opt);
}

simaai::neat::RunOptions build_run_options() {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  return run_options;
}

std::string stream_id_for(int stream_index) {
  return "stream" + std::to_string(stream_index);
}

int stream_index_from_sample(const simaai::neat::Sample& sample, int stream_count) {
  const std::string prefix = "stream";
  if (sample.stream_id.rfind(prefix, 0) != 0) {
    if (stream_count == 1) {
      return 0;
    }
    throw std::runtime_error("pose sample missing stream id: " + sample.stream_id);
  }
  const std::string suffix = sample.stream_id.substr(prefix.size());
  if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(),
                                     [](unsigned char c) { return std::isdigit(c) != 0; })) {
    throw std::runtime_error("invalid pose stream id: " + sample.stream_id);
  }
  const int index = std::stoi(suffix);
  if (index < 0 || index >= stream_count) {
    throw std::runtime_error("pose stream id out of range: " + sample.stream_id);
  }
  return index;
}

simaai::neat::GraphLinkOptions realtime_link(int stream_index, int max_inflight_per_stream = -1,
                                             int max_inflight_total = -1) {
  simaai::neat::GraphLinkOptions link;
  link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.max_inflight_per_stream = max_inflight_per_stream;
  link.max_inflight_total = max_inflight_total;
  link.stream_id = stream_id_for(stream_index);
  return link;
}

simaai::neat::nodes::groups::VideoSenderOptions make_video_options(const AppConfig& cfg,
                                                                   int stream_index) {
  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(cfg.codec);
  video_options.host = cfg.insight_host;
  video_options.channel = stream_index;
  video_options.video_port_base = cfg.video_port_base;
  video_options.async = true;
  return video_options;
}

StreamRuntime build_stream_runtime(const AppConfig& cfg, int stream_index, const std::string& url) {
  StreamRuntime runtime;
  runtime.index = stream_index;
  int output_fps = 0;
  const auto source_options =
      build_source_options(cfg, url, output_fps, runtime.frame_w, runtime.frame_h);
  sima_examples::require(runtime.frame_w > 0 && runtime.frame_h > 0,
                         "failed to probe RTSP frame dimensions");
  sima_examples::require(output_fps > 0, "failed to probe RTSP frame rate");

  runtime.profile.enabled = cfg.profile;
  runtime.profile.stream_index = stream_index;
  runtime.source_options = source_options;
  runtime.tracker = multi_stream_people_tracker::PeopleTracker(
      cfg.tracking_iou_threshold, cfg.tracking_max_missing_frames);
  runtime.fps = output_fps;
  int video_port = 0;
  if (cfg.video_enabled) {
    video_port = make_video_options(cfg, stream_index).video_port();
  }

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = stream_index;
  metadata_options.metadata_port_base = cfg.metadata_port_base;
  std::string metadata_err;
  runtime.metadata_sender =
      std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(runtime.metadata_sender->ok(), metadata_err);

  std::cout << "[stream " << stream_index << "] rtsp=" << url << " stream=" << runtime.frame_w
            << "x" << runtime.frame_h << "@" << output_fps << " insight=" << cfg.insight_host
            << " video=";
  if (cfg.video_enabled) {
    std::cout << video_port;
  } else {
    std::cout << "disabled";
  }
  std::cout << " metadata=" << runtime.metadata_sender->metadata_port() << "\n";
  return runtime;
}

void connect_stream_graph(AppRuntime& app, const AppConfig& cfg, const StreamRuntime& stream,
                          const simaai::neat::Graph& estimator_graph,
                          const simaai::neat::Graph* frame_graph) {
  auto source = build_encoded_source_graph(stream.source_options);
  auto decoder = build_decode_graph("decode_h264", stream.source_options);

  if (cfg.video_enabled) {
    auto encoded_branch = simaai::neat::graphs::Branch("encoded", {"decode_h264", "video_h264"});
    app.graph.connect(source, encoded_branch);
    app.graph.connect(encoded_branch, decoder, realtime_link(stream.index));

    const auto video_options = make_video_options(cfg, stream.index);
    app.graph.connect(encoded_branch,
                      build_video_sender_graph("video_h264", cfg.codec, video_options),
                      realtime_link(stream.index));
  } else {
    app.graph.connect(source, decoder, realtime_link(stream.index));
  }

  auto decoded_branch = frame_graph
                            ? simaai::neat::graphs::Branch("decoded", {"estimator_frame", "frame"})
                            : simaai::neat::graphs::Branch("decoded", {"estimator_frame"});
  app.graph.connect(decoder, decoded_branch);
  app.graph.connect(
      decoded_branch, estimator_graph,
      realtime_link(stream.index, cfg.max_inflight_per_stream, cfg.max_inflight_total));
  if (frame_graph) {
    app.graph.connect(decoded_branch, *frame_graph, realtime_link(stream.index));
  }
}

void send_metadata(StreamRuntime& stream, const AppConfig& cfg, const simaai::neat::Sample& sample,
                   const std::vector<TrackedPose>& tracked_poses) {
  const std::string pose_json = pose_metadata_data_json(cfg, tracked_poses);
  const std::string tracking_json = tracking_metadata_data_json(cfg, tracked_poses);
  const int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  if (!stream.metadata_sender->send_metadata("pose-estimation", pose_json, timestamp_ms, frame_id,
                                             &err)) {
    std::cerr << "[warn] stream " << stream.index << " pose metadata send failed: " << err << "\n";
  }
  if (!stream.metadata_sender->send_metadata("tracking", tracking_json, timestamp_ms, frame_id,
                                             &err)) {
    std::cerr << "[warn] stream " << stream.index << " tracking metadata send failed: " << err
              << "\n";
  }
}

bool all_streams_done(const std::vector<StreamRuntime>& streams, int frame_limit) {
  if (frame_limit <= 0) {
    return false;
  }
  return std::all_of(streams.begin(), streams.end(), [frame_limit](const StreamRuntime& stream) {
    return stream.processed >= frame_limit;
  });
}

void process_output_sample(StreamRuntime& stream, AppRuntime& app, const AppConfig& cfg,
                           const simaai::neat::Sample& sample, double detection_pull_ms) {
  if (cfg.frames > 0 && stream.processed >= cfg.frames) {
    return;
  }

  const auto& pose_sample = pose_sample_from_output(cfg, sample);
  const auto poses = decode_poses(pose_sample, stream.frame_w, stream.frame_h, cfg.max_poses);

  ++stream.processed;
  const bool warming_up = stream.processed <= cfg.warmup_frames;
  if (!warming_up) {
    std::vector<multi_stream_people_tracker::Detection> detections;
    detections.reserve(poses.size());
    for (const auto& pose : poses) {
      detections.push_back({pose.x1, pose.y1, pose.x2, pose.y2, pose.score, 0});
    }
    const auto tracked =
        stream.tracker.update(detections, stream.processed);
    std::vector<int> active_track_ids;
    std::vector<TrackedPose> tracked_poses;
    tracked_poses.reserve(tracked.size());
    active_track_ids.reserve(tracked.size());

    const double timestamp_s =
        sample.pts_ns >= 0
            ? static_cast<double>(sample.pts_ns) / 1'000'000'000.0
            : static_cast<double>(stream.processed) / static_cast<double>(std::max(1, stream.fps));
    for (std::size_t i = 0; i < tracked.size() && i < poses.size(); ++i) {
      const int track_id = tracked[i].track_id;
      active_track_ids.push_back(track_id);
      TrackedPose tracked_pose =
          stream.fall_detector.update(track_id, poses[i], timestamp_s, stream.frame_h, cfg);
      if (tracked_pose.confirmed_now && app.alert_sink) {
        std::optional<std::string> description;
        if (app.vlm_router) {
          description = app.vlm_router->describe_confirmed_fall(sample, tracked_pose);
        }
        app.alert_sink->emit_fall(timestamp_s, track_id, tracked_pose.confidence, description);
      }
      tracked_poses.push_back(std::move(tracked_pose));
    }
    stream.fall_detector.retain_only(active_track_ids, cfg.tracking_max_missing_frames);

    const double metadata_start = sima_examples::time_ms();
    send_metadata(stream, cfg, pose_sample, tracked_poses);
    const double metadata_end = sima_examples::time_ms();
    stream.profile.add(detection_pull_ms, metadata_end - metadata_start,
                       static_cast<int>(tracked_poses.size()));
    app.performance.add(detection_pull_ms);
    if (app.telemetry) {
      app.telemetry->publish(cfg, stream, timestamp_s, detection_pull_ms,
                             metadata_end - metadata_start, tracked_poses);
    }
  }
}

void run_app(const AppConfig& cfg) {
  g_stop_requested = 0;
  auto previous_sigint = std::signal(SIGINT, request_stop);

  AppRuntime app;
  app.performance.vlm_enabled = cfg.vlm_enabled;
  app.alert_sink = std::make_unique<AlertSink>(cfg);
  app.telemetry = std::make_unique<TelemetryPublisher>(cfg);
  if (cfg.vlm_enabled) {
    app.vlm_router = std::make_unique<VlmRouter>(cfg);
  }
  app.streams.reserve(cfg.rtsp_urls.size());
  std::cout << "HomeAngel AI using YOLO26 pose model: " << cfg.model_path << "\n";
  std::cout << "Privacy boundary: video stays in memory/local Insight; JSON events append to "
            << cfg.events_log << "\n";
  if (cfg.vlm_enabled) {
    std::cout << "VLM descriptions: enabled, local endpoint http://" << cfg.vlm_host << ":"
              << cfg.vlm_port << ", model route=";
    for (std::size_t i = 0; i < cfg.vlm_models.size(); ++i) {
      if (i != 0) {
        std::cout << " -> ";
      }
      std::cout << cfg.vlm_models[i];
    }
    std::cout << "\n";
  } else {
    std::cout << "VLM descriptions: disabled (set vlm.enabled=true after starting local GenAI)\n";
  }
  // One model, shared by every stream: the estimator graph is built once here and each
  // stream's decoded branch links into it below.
  app.model = build_model(cfg);
  auto input_options = app.model->input_appsrc_options(false);
  input_options.block = true;
  simaai::neat::Graph estimator_graph("estimator");
  estimator_graph.connect(simaai::neat::nodes::Input("estimator_frame", input_options), *app.model);

  simaai::neat::Graph poses_graph("poses");
  poses_graph.add(simaai::neat::nodes::Output("poses", simaai::neat::OutputOptions::EveryFrame(4)));

  std::optional<simaai::neat::Graph> frame_graph;
  if (cfg.vlm_enabled) {
    frame_graph.emplace("frame");
    frame_graph->add(
        simaai::neat::nodes::Output("frame", simaai::neat::OutputOptions::EveryFrame(4)));
  }

  for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
    app.streams.push_back(build_stream_runtime(cfg, static_cast<int>(index), cfg.rtsp_urls[index]));
    connect_stream_graph(app, cfg, app.streams.back(), estimator_graph,
                         frame_graph ? &*frame_graph : nullptr);
  }
  app.graph.connect(estimator_graph, poses_graph);
  if (frame_graph) {
    auto joined = simaai::neat::graphs::Combine({"frame", "poses"}, "analysis_output",
                                                simaai::neat::CombinePolicy::ByFrame);
    app.graph.connect(*frame_graph, joined);
    app.graph.connect(poses_graph, joined);
    app.output_name = "analysis_output";
  }

  if (cfg.profile) {
    std::cout << "Backend:\n" << app.graph.describe_backend() << "\n";
  }

  app.run = app.graph.build(build_run_options());
  while (g_stop_requested == 0 && !all_streams_done(app.streams, cfg.frames)) {
    constexpr int kPullTimeoutMs = 50;
    const double pull_start = sima_examples::time_ms();
    simaai::neat::Sample sample;
    simaai::neat::PullError pull_error;
    const auto status = app.run.pull(app.output_name, kPullTimeoutMs, sample, &pull_error);
    const double pull_end = sima_examples::time_ms();
    if (status == simaai::neat::PullStatus::Timeout || status == simaai::neat::PullStatus::Closed) {
      continue;
    }
    if (status != simaai::neat::PullStatus::Ok) {
      throw std::runtime_error("failed to pull " + app.output_name + ": " + pull_error.message);
    }
    const int stream_index =
        stream_index_from_sample(pose_sample_from_output(cfg, sample),
                                 static_cast<int>(app.streams.size()));
    process_output_sample(app.streams[static_cast<std::size_t>(stream_index)], app, cfg, sample,
                          pull_end - pull_start);
  }
  app.run.close();

  for (auto& stream : app.streams) {
    stream.profile.flush();
    std::cout << "[stream " << stream.index << "] processed=" << stream.processed << "\n";
  }
  app.performance.print();
  std::signal(SIGINT, previous_sigint);
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    if (!fs::exists(cli.config_path)) {
      std::cerr << "Error: config file not found: " << cli.config_path << "\n";
      return 2;
    }

    const AppConfig cfg = load_app_config(cli.config_path);
    if (cli.validate_config_only) {
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.rtsp_urls.size()
                << ", max_inflight_per_stream=" << cfg.max_inflight_per_stream
                << ", max_inflight_total=" << cfg.max_inflight_total << ")\n";
      return 0;
    }
    run_app(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
