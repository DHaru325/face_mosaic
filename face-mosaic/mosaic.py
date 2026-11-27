import cv2
import numpy as np  


img = cv2.imread("input.png")

if img is None:
    print("이미지가 없습니다. 파일 이름을 확인하세요.")
    exit()


face_xml = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_xml.detectMultiScale(gray_img, 1.1, 5)
print("찾은 얼굴 개수:", len(faces))


for (x, y, w, h) in faces:
    
    
    face_part = img[y : y + h, x : x + w]

    
    mosaic_level = 15

    
    small_w = int(w / mosaic_level)
    small_h = int(h / mosaic_level)
    
    
    if small_w == 0:
        small_w = 1
    if small_h == 0:
        small_h = 1

    small_face = cv2.resize(face_part, (small_w, small_h))

    
    big_face = cv2.resize(
        small_face, (w, h), interpolation=cv2.INTER_NEAREST
    )

    
    img[y : y + h, x : x + w] = big_face


cv2.imwrite("mosaic_result.jpg", img)
print("저장 완료")


cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
