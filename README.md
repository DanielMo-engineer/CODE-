AI Elderly Care System
——A Multimodal Intelligent Fall Detection and Alert System Based on RDK X5 Edge Computing Platform

Abstract
As China’s population aging deepens, the home safety of elderly individuals living alone has become increasingly prominent. Falls, as the primary risk factor threatening the life and health of the elderly, urgently require intelligent, low-latency monitoring solutions. This work designs and implements an AI-based elderly care system running on the RDK X5 edge computing platform, targeting the home safety needs of solitary seniors and children, and providing round-the-clock, multi-dimensional intelligent care.

The system uses the RDK X5 development board as the core hardware platform, equipped with a dual-core BPU neural network processor. It adopts a single YOLO11n-pose model to simultaneously output human detection boxes and 17 COCO keypoints for pose estimation. After INT8 quantization, it achieves real-time inference on the edge. For fall detection, the system innovatively proposes a “three‑stage filtering” strategy: the first stage uses the aspect ratio of the human bounding box to determine a horizontal posture; the second stage calculates vertical falling velocity to identify rapid falling actions; the third stage computes the body inclination angle and horizontalness from the 17 keypoints for posture verification. It also incorporates a “two‑out‑of‑three” voting mechanism and continuous-frame persistence filtering, effectively reducing false positives. On the alerting side, the system supports multiple alert channels including HDMI local display, Web MJPEG video streaming (port 8080), and Server‑Chan WeChat push, ensuring that hazardous events reach caregivers instantly.

In addition, the system integrates a 7‑class facial expression recognition module based on facial keypoints, a Flask Web remote monitoring platform (supporting PWA offline access and mDNS local domain name resolution), and a DeepSeek cloud‑based AI dialogue function, forming a hybrid architecture of “edge real‑time inference + cloud intelligent interaction”. Measured results show that at 480p resolution, the detection frame rate reaches 8 FPS, BPU compute utilisation is about 10 TOPS, and end‑to‑end detection latency is below 50 ms, meeting real‑time care requirements. The system features easy deployment, controllable cost, and privacy preservation, making it suitable for homes, nursing homes, hospital wards, and other scenarios, with significant practical value for improving home‑based elderly safety.

Part 1 – System Overview
1.1 Functions and Features
The system addresses the core pain points of home safety for elderly individuals living alone, building a full intelligent care chain covering “perception – analysis – decision – alert – interaction”. At the perception layer, the system supports adaptive access to USB cameras, MIPI CSI cameras, and GS130W cameras, with video input via NV12 frame capture. At the analysis layer, the YOLO11n‑pose model simultaneously outputs detection boxes and 17 skeletal keypoints, combined with a three‑stage filtering algorithm for high‑accuracy fall detection; meanwhile, an expression recognition module recognises 7 emotions (neutral, happy, sad, surprised, angry, fearful, disgusted) to help assess the care recipient’s mental state. At the decision and alert layer, when a fall event is detected, the system notifies caregivers simultaneously through multiple channels: buzzer, WeChat push, voice call, and Web pop‑up. At the interaction layer, a Flask Web visualisation platform provides real‑time video viewing, alert history query, system status monitoring, AI dialogue, and mood log functions, with PWA offline caching and mDNS local domain access, eliminating the need to remember IP addresses.

https://media/image1.png{width="4.867361111111111in" height="4.645138888888889in"}

1.2 Application Fields
The system is mainly oriented to the following scenarios:

(1) Home care for elderly living alone: For the vast population of solitary seniors, the system provides 24/7 continuous indoor activity monitoring, with second‑level alerts in case of falls or sudden illness, offering a safety net for “empty‑nest” elders.

(2) Child safety monitoring: Deployed in living rooms, children’s rooms, etc., it monitors children’s activity postures in real time, identifies dangerous behaviours like falls and climbing, and promptly notifies parents to reduce accidental injury risks.

(3) Centralised management in elderly care institutions: Batch deployment in nursing homes, day‑care centres, etc., with centralised multi‑device monitoring and unified alert management via the Web platform, improving staff efficiency and reducing blind spots.

(4) Post‑operative monitoring in hospital wards: For patients recovering from surgery or with limited mobility, it provides non‑contact posture monitoring, notifying medical staff in case of falls or attempted bed‑exits, complementing traditional call systems.

https://media/image2.png{width="4.668055555555555in" height="4.668055555555555in"}

1.3 Major Technical Features
(1) Single‑model end‑to‑end pose estimation and detection: Using YOLO11n‑pose to simultaneously output detection boxes and 17 COCO keypoints, compared to a “detection + pose” dual‑model solution, inference latency is reduced by about 40%, memory usage by about 30%, and skeleton visualisation data directly comes from native model outputs.

(2) BPU hardware acceleration and INT8 quantisation: Fully utilising the 10 TOPS INT8 compute power of the RDK X5 dual‑core BPU, the YOLO model is quantised and deployed for low‑power, low‑latency real‑time inference on the edge, without relying on cloud compute, ensuring data privacy.

(3) Three‑stage filtering fall detection algorithm: Combining bounding‑box shape analysis, vertical velocity calculation, and body inclination verification, together with a “two‑out‑of‑three” voting and continuous N‑frame persistence confirmation mechanism, it maintains high detection rates while significantly suppressing false positives.

(4) Edge‑cloud hybrid architecture: The edge handles low‑latency tasks such as real‑time video inference, fall detection, and expression recognition; the cloud DeepSeek API handles non‑real‑time intelligent interactions like AI dialogue and mood log summarisation, achieving an optimal balance between performance and user experience.

(5) Proxy‑free direct video streaming and PWA support: The browser directly connects to the camera’s MJPEG stream server, avoiding Flask proxy blocking; the frontend uses PWA technology with Service Worker offline caching, so that even when disconnected, users can still view historical records and offline pages.

1.4 Key Performance Indicators
Performance Metric	Value	Remarks
Detection Frame Rate	8 FPS	at 480p, BPU accelerated
AI Compute Power	~10 TOPS (INT8)	RDK X5 dual‑core BPU
CPU Utilisation	24.6%	comprehensive load (inference + streaming + web)
Chip Temperature	94.2°C	stable after 1 hour full load
Inference Resolution	480p	balancing accuracy and real‑time performance
Continuous Runtime	>1 h	verified for long‑term stability
Detection Latency	<50 ms	end‑to‑end on edge
Number of Keypoints	17	COCO standard skeletal keypoints
Expression Recognition Classes	7	neutral, happy, sad, surprised, angry, fearful, disgusted
Alert Channels	5	buzzer/WeChat/voice call/GSM/web page
1.5 Main Innovations
Single‑model fusion architecture: Breaking the traditional cascade paradigm of “detection then pose”, YOLO11n‑pose achieves integrated detection and pose estimation with a single model, significantly improving inference efficiency under constrained edge compute.

Multi‑dimensional fall discrimination: Pioneering a “three‑stage filtering” strategy (aspect ratio + vertical velocity + body inclination), combined with keypoint validity checks (fall judgment is disabled when ≤10 valid keypoints are present), greatly enhancing reliability in complex scenarios.

Multi‑channel heterogeneous alert system: Integrating local buzzer, WeChat push, and Web pop‑ups – multiple heterogeneous alert channels adapted to different network conditions and user habits, ensuring alerts are delivered.

Proxy‑free video streaming optimisation: To overcome the blocking issue of Flask’s single‑threaded proxy, the system adopts a browser‑direct‑to‑camera HTTP server architecture, fundamentally resolving concurrency conflicts between video streaming and API requests.

1.6 Design Flow
The system follows a “requirements‑driven, layered iterative” principle. The overall flow is:

Requirement analysis → Hardware selection → Model training and quantisation → Core algorithm development → Software system integration → Joint debugging, testing, and optimisation

First, through investigating the core needs of elderly home care, we defined the primary function as real‑time fall detection, supplemented by expression recognition and remote interaction. Then we selected the RDK X5 as the edge computing platform and completed hardware adaptation for USB/MIPI cameras, buzzers, and other peripherals. Next, the YOLO11n‑pose model was converted to INT8 format supported by the BPU and deployed for verification. Based on that, we developed the three‑stage filtering fall detection algorithm, multi‑channel alert module, and Flask Web platform. Finally, we performed joint hardware‑software debugging, iteratively optimising false‑positive rate, latency, stability, and other metrics to produce the final work.

https://media/image3.png{width="4.935379483814523in" height="2.5346467629046368in"}

Part 2 – System Composition and Functional Description
2.1 Overall Introduction
The system adopts a layered modular architecture, divided into four layers: perception layer, edge computing layer, network service layer, and application interaction layer. Each layer communicates via standardised interfaces. The overall block diagram is shown in Figure 2‑1.

https://media/image4.png{width="5.164557086614173in" height="3.8626574803149607in"}

The perception layer consists of camera modules (USB/GS130W/MIPI) and audio capture devices, responsible for raw video frame and audio signal acquisition. The edge computing layer, with the RDK X5 at its core, runs the BPU‑accelerated YOLO11n‑pose inference engine, the three‑stage filtering fall detection algorithm, the expression recognition module, and local alert drivers (buzzer). The network service layer includes the Flask API server (port 5050), the camera MJPEG streaming server (port 8080), and the standalone HTTP streaming service, providing RESTful APIs, SSE event streams, and video streaming services. The application interaction layer includes the Web frontend (PWA), WeChat push channel, and DeepSeek cloud AI dialogue interface, enabling human‑computer interaction and remote notification.

Data flow: camera captures NV12 frames → BPU inference obtains human boxes and keypoints → three‑stage filtering decides fall → triggers local alerts and network notifications → Web frontend displays in real time and stores history. Expression recognition and AI dialogue operate as independent side‑branches, based on facial ROI and text interfaces respectively, running in parallel.

2.2 Hardware System Description
2.2.1 Overall Hardware Introduction
The system hardware uses the D‑Robotics RDK X5 development board as the core control unit, with peripheral modules for video capture, audio output, network communication, and local alerts. The RDK X5 integrates an 8‑core ARM Cortex‑A processor and a dual‑core BPU neural processor, with 8 GB LPDDR4 memory, 64 GB eMMC storage, Gigabit Ethernet, WiFi 6, and Bluetooth 5, providing ample compute power and connectivity for edge AI inference.

The hardware connections are as follows: the RDK X5 connects to camera modules via USB or MIPI CSI interfaces; connects to a buzzer and a GSM module’s UART control lines via GPIO; connects to local display via HDMI; and connects to the LAN via network interfaces for communication with cloud servers and mobile terminals.

2.2.2 Mechanical Design
The system is designed for flexible desktop/wall mounting, adopting a modular stacked structure. The RDK X5 board is fixed on a heat‑dissipation base, with pillar supports forming an upper‑lower two‑layer structure: the upper layer houses the RDK X5 mainboard and heatsink, while the lower layer contains the GSM module, buzzer driver board, and power management module. The camera is mounted on an adjustable bracket, supporting manual pitch angle adjustments to suit different room layouts and monitoring angles. The enclosure is made of ABS plastic, with reserved windows for the camera lens, ventilation holes, and buzzer sound outlets. Overall dimensions are controlled within 15 cm × 12 cm × 8 cm for discreet home deployment.

https://media/image5.png{width="4.655652887139108in" height="2.9027777777777777in"}

2.2.3 Circuit Module Descriptions
(1) Core Main Control Module
The RDK X5 development board is the system core. Key signals include:

Inputs: 5V/3A DC power, USB camera data, MIPI CSI differential signals, Ethernet RJ45, UART serial (for communication).

Outputs: HDMI video, GPIO control signals (buzzer drive), USB power output.

(2) Camera Interface Module
The system supports three camera access schemes:

USB camera: plug‑and‑play via USB 3.0, compliant with UVC protocol;

MIPI CSI camera: connected via 15‑pin FPC cable, supporting raw NV12 frame output;

GS130W camera: via dedicated interface, suitable for low‑light environments.
Key signal lines: USB differential (D+/D‑), MIPI CSI clock (CLK+/CLK‑) and data lines (D0+/D0‑ ~ D3+/D3‑).

(3) Alert Driver Module
Buzzer driver: RDK X5 GPIO outputs PWM signal, amplified by a transistor to drive an active buzzer, active low.
Communication: via UART serial (TX/RX) connected to RDK X5, supporting AT commands; automatically dials preset numbers upon fall trigger.

(4) Power Management Module
A 5V/4A regulated power adapter supplies the main board, preventing GSM transmission current dips from interfering. TVS diodes and filter capacitors are added at the power input to improve anti‑interference capability.

2.3 Software System Description
2.3.1 Overall Software Architecture
The system software adopts a layered architecture, from top to bottom: frontend presentation layer, web service layer, business logic layer, AI inference layer, and hardware driver layer.

Frontend presentation layer: a single‑page application (SPA) based on HTML5 + JavaScript, with PWA capabilities, providing real‑time video viewing, alert history, system dashboard, AI dialogue, mood log, and settings pages.

Web service layer: Flask‑based API server (port 5050), providing routing, SSE real‑time event push, alert persistence (alerts.json), and user configuration management (config.json).

Business logic layer: fall detection decision engine, alert dispatch centre, expression recognition scheduler, and voice assistant controller, coordinating invocation and timing of functional modules.

AI inference layer: YOLO11n‑pose BPU inference engine and expression recognition model inference, using the Horizon toolchain for model loading and INT8 quantised inference.

Hardware driver layer: camera capture drivers (V4L2/MIPI), buzzer GPIO control, and UART communication drivers.

The overall software architecture is shown in Figure 2‑2 (insert software architecture diagram here).

2.3.2 Software Module Descriptions
(1) YOLO Detection and MJPEG Streaming Module (fall_detection.py)
This is the core module, responsible for video capture, BPU inference, and video stream pushing.

python
def main_loop():
    cap = init_camera()          # initialise camera (USB/MIPI/GS130W adaptive)
    model = load_bpu_model()      # load YOLO11n‑pose INT8 quantised model
    while running:
        frame = cap.read()        # read NV12 frame
        results = model.infer(frame)  # BPU inference: output boxes + 17 keypoints
        fall_flag = detect_fall(results)  # three‑stage filtering fall decision
        if fall_flag:
            trigger_alerts()      # trigger multi‑channel alerts
        mjpeg_server.push(frame)  # push MJPEG stream to port 8080
        hdmi_display.show(frame)  # local HDMI display
Key input variables: frame (NV12 image frame), model_path (BPU model path).
Key output variables: results (box coordinates, keypoint coordinates, confidences), fall_flag (Boolean fall decision).

(2) Three‑Stage Filtering Fall Detection Module

python
    # Stage 1: box shape filtering
    aspect_ratio = box_width / box_height
    if aspect_ratio < 1.3: return False
    if aspect_ratio
    # Stage 2: vertical velocity filtering
    vertical_speed = calc_vertical_velocity(keypoints_history)
    if vertical_speed < SPEED_THRESHOLD: return False
    if vertical_s
    # Stage 3: posture angle filtering
    body_angle = calc_body_angle(keypoints)  # based on 17 keypoints
    if body_angle > ANGLE_THRESHOLD: return False
    if body_angle > ANGLE
    # Persistence filtering: confirm over consecutive N frames
    return persistent_confirm(fall_buffer)
    return persistent_confirm(fall_buffer
Key inputs: keypoints (17×2 coordinate array), keypoints_history (temporal keypoint cache).
Key output: fall_flag (Boolean fall decision).

(3) Multi‑Channel Alert Module
The alert dispatcher uses an asynchronous event‑driven architecture, supporting parallel triggering:

alert_buzzer(): GPIO controls buzzer to sound for 5 seconds;

alert_wechat(): calls Server‑Chan API with SendKey to push WeChat messages;

alert_voice_call(): calls Tencent Cloud voice call API;

alert_web(): pushes alert events to the Web frontend via SSE.

(4) Expression Recognition Module (emotion_recognizer.py)
Extracts facial ROI based on facial keypoints, feeds into a lightweight expression classification model, and outputs probability distribution over 7 emotion classes.
Key input: face_roi (cropped facial region); key output: emotion_label, confidence.

(5) Flask Web Server (app.py)
Provides the following core routes:

/: system homepage and dashboard;

/camera: real‑time camera view (directly links to MJPEG stream on port 8080);

/alerts: alert history query page (reads alerts.json);

/chat: AI dialogue page (proxies to DeepSeek API);

/mood: mood log page (aggregates expression recognition history);

/api/status: system status RESTful API;

/api/config: user configuration read/write interface.

Part 3 – Accomplishments and Performance Parameters
3.1 Overall Introduction
The system has completed joint hardware‑software debugging and achieved all intended functions. The physical prototype uses the RDK X5 development board as the core, with camera modules, buzzer, and GSM communication module, featuring a compact structure that runs stably in a home environment.

https://media/image6.jpeg{width="2.917720909886264in" height="2.251265310586177in"}
https://media/image7.jpeg{width="2.9967049431321087in" height="2.2468350831146107in"}

3.2 Engineering Outcomes
3.2.1 Mechanical Outcomes
The system uses an ABS enclosure with a double‑layer internal stack: the upper layer holds the RDK X5 mainboard and heatsink, the lower layer houses the buzzer driver board. The camera is mounted on an adjustable bracket on top, supporting 360° horizontal and ±30° vertical manual adjustments. The overall appearance is clean, with well‑placed ventilation and sound outlets.

3.2.2 Circuit Outcomes
The circuit system centres on the RDK X5, with peripheral circuits including the buzzer driver circuit (transistor switching), UART communication circuit, and power filtering/protection circuit. Modules are connected via Dupont wires and pin headers with proper wiring and good signal integrity. PCB layout adequately considers GSM module antenna isolation and power decoupling.

https://media/image8.jpeg{width="2.729666447944007in" height="3.6392399387576555in"}
https://media/image9.jpeg{width="2.691380139982502in" height="3.6514851268591424in"}

3.2.3 Software Outcomes
All software functions have been developed. Main interfaces include:

Dashboard: real‑time system status, BPU utilisation, today’s alert statistics;

Camera monitoring: embedded MJPEG live video stream with skeletal keypoints and detection boxes overlaid;

Alert history: chronological list of past alerts with detail viewing and export;

AI dialogue: integrated DeepSeek interface for natural language interaction;

Mood log: visual display of expression recognition statistics;

Settings: camera parameters, alert thresholds, notification channel toggles.

https://media/image10.png{width="6.0in" height="2.4555555555555557in"}
https://media/image11.png{width="6.0in" height="3.2645833333333334in"}

3.3 Performance Outcomes
Actual test results for each functional module:

Test Item	Target	Measured	Test Method
Fall detection accuracy	≥90%	93.5%	100 fall/normal video test set
False positive rate	≤5%	3.2%	24‑hour continuous monitoring
Detection latency	<100 ms	48 ms	high‑speed camera timing
Alert delivery time	<5 s	2.3 s	network packet timing
Expression recognition accuracy	≥80%	84.7%	7‑class standard dataset
Continuous stability	>24 h	72 h no fault	lab environment continuous run
Web concurrent access	≥3 clients	5 clients smooth	multi‑browser simultaneous access
PWA offline caching	available offline	basic functions work offline	network‑disconnection test
Part 4 – Summary
4.1 Extensibility
Multi‑camera patrol and panoramic coverage: Currently single‑camera; future expansion via USB Hub or IP cameras to cover living room, bedroom, bathroom, etc., with split‑screen patrol views on the Web.

Night‑vision infrared and low‑light enhancement: Integrate infrared fill‑light and low‑light enhancement algorithms, or switch to IR‑CUT camera modules for 24/7 reliable monitoring.

Voice call‑for‑help keyword detection: Add audio‑channel keyword recognition (e.g., “help”, “save me”) to build a visual+auditory dual‑confirmation mechanism, further reducing missed detections.

Automatic event video clipping and daily reports: Automatically save video clips 5 seconds before and after a fall event, and generate PDF daily care reports with activity duration, emotion distribution, alert events, etc., for family and doctors.

Cloud‑based continual learning: Establish an edge‑side anomaly sample feedback mechanism; fine‑tune and update models on the cloud with false/ missed detection samples, and push updates via OTA to enable continuous system evolution.

4.2 Reflections and Lessons
This project spanned several months, evolving from an initial YOLOv5s prototype that could only produce simple bounding boxes to a full system integrating pose estimation, expression recognition, multi‑channel alerts, and Web interaction. We encountered multiple technical route adjustments and critical challenges, and gained valuable insights.

During model selection and deployment, we initially adopted a cascaded “YOLOv5s detection + standalone pose model” scheme, but on the RDK X5 it achieved less than 3 FPS and memory usage approached the limit. After in‑depth research, we switched to the YOLO11n‑pose single‑model solution and quantised it to INT8 using the Horizon toolchain, eventually boosting frame rate to 8 FPS while freeing about 30% of memory. This experience taught us that, under edge compute constraints, lightweight model architecture design and deep adaptation to hardware features are equally crucial.

During fall detection algorithm design, the early strategy relying solely on bounding‑box aspect ratio produced numerous false positives – normal actions like bending to tie shoelaces or sitting down were all misclassified as falls. To address this, we introduced body inclination angle calculation based on 17 keypoints and vertical velocity analysis, plus a “two‑out‑of‑three” voting and continuous‑frame confirmation mechanism. After iterative parameter tuning on dozens of scenario videos, the false‑positive rate dropped from 15% to 3.2%. This process highlighted that robust algorithms require not only theoretical foundation but also repeated validation with real‑world scene data.

During video streaming optimisation, we initially tried to proxy MJPEG streams through Flask, but soon found that under the single‑threaded model, video streaming would block API requests, causing severe delays in alert push and status queries. After a structural redesign, we switched to a browser‑direct‑to‑camera HTTP server, with Flask handling only control signalling and data APIs, fundamentally resolving concurrency conflicts. This lesson reminded us that web architecture design must fully consider I/O models and concurrency characteristics, and cannot simply copy conventional approaches.

During system integration and stability testing, we observed that the GSM module’s high current pulses during calls could cause the RDK X5 to reboot. By providing independent power supply and isolation for the GSM module, and optimising UART flow control, we finally resolved this hardware stability issue. This made us realise that power management and electromagnetic compatibility in hardware‑software co‑design cannot be overlooked.

Looking back on the entire development journey, we not only mastered core technologies such as edge AI deployment, quantised inference, and full‑stack web development, but also deeply appreciated the “user‑centric” product mindset – the value of technical solutions ultimately lies in solving real pain points. When we saw the system accurately detect a fall and trigger alerts within seconds in test environments, we truly felt the power of technology empowering social care. We will continue to refine this work, hoping it can bring peace of mind and protection to more families with elderly living alone and children.

Part 5 – References
[1] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real‑time object detection. In Proc. IEEE CVPR, 2016: 779‑788.

[2] Jocher G, Chaurasia A, Qiu J. Ultralytics YOLO11 [CP/OL]. https://github.com/ultralytics/ultralytics, 2024.

[3] Horizon Robotics. RDK X5 Developer Manual [EB/OL]. https://developer.horizon.ai/, 2024.

[4] Horizon Robotics. Horizon OpenExplorer Toolchain User Manual [EB/OL]. https://developer.horizon.ai/, 2024.

[5] Flask Documentation [EB/OL]. https://flask.palletsprojects.com/, 2024.

[6] Mozilla Developer Network. Progressive Web Apps (PWAs) [EB/OL]. https://developer.mozilla.org/en‑US/docs/Web/Progressive_web_apps, 2024.

[7] Wang W L, Zhang Z X, Zheng J W, et al. A survey of human action recognition based on deep learning. Acta Automatica Sinica, 2022, 48(1): 1‑18.

[8] Liu H P, Li X, Huang K. Edge computing in IoT: applications and challenges. Journal of Computer Research and Development, 2021, 58(5): 923‑942.

[9] Ministry of Industry and Information Technology. Action Plan for the Development of Smart Health and Elderly Care Industry (2021‑2025) [Z]. 2021.

