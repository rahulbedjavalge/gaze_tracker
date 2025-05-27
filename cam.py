import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret:
        print(f"Camera index {i} is working")
        cv2.imshow(f'Camera {i}', frame)
        cv2.waitKey(1000)  # show for 1 second
    cap.release()

cv2.destroyAllWindows()
