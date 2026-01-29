import cv2

cap = cv2.VideoCapture(0)

avg_bg = None
 

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    #Convert colors in grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if avg_bg is None:
        avg_bg = blur.copy().astype("float")
        continue
    
    cv2.accumulateWeighted(gray, avg_bg, 0.02)

    ref_frame = cv2.convertScaleAbs(avg_bg)

    #Calcule the diff between previous frame and atual
    diff = cv2.absdiff(gray, ref_frame)


    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    thresh = cv2.dilate(thresh, None, iterations=2)


    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < 4000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    prev_frame = gray
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()