import cv2

video_path = "машинки_врум_врум.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
MIN_AREA = 2000

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    roi = frame[height // 4:, :]

    fg_mask = bg_subtractor.apply(roi)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        y += height // 4 

        if y + h < height // 3:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    max_width = 1200
    scale = min(max_width / frame.shape[1], 1.0)
    display_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame
    display_mask = cv2.resize(fg_mask, (display_frame.shape[1], display_frame.shape[0]))

    cv2.imshow("Cars Detection", display_frame)
    cv2.imshow("Motion Mask", display_mask)  

    if cv2.waitKey(30) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
