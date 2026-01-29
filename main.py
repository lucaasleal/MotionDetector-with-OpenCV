import cv2

cap = cv2.VideoCapture(0)

prev_frame = None
 

while True:
    ret, frame = cap.read()

    if not ret:
        break

    #Convert colors in grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray
        continue

    #Calcule the diff between previous frame and atual
    diff = cv2.absdiff(prev_frame, gray)

    blur = cv2.GaussianBlur(diff, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh, None, iterations=2)


    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Difference", diff)
    cv2.imshow("Threshold", thresh)

    prev_frame = gray
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()