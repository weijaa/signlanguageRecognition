import cv2
import os, time
for i in range(1):
    startTime=0.0
    end = 0.0
    cap = cv2.VideoCapture(0)
    cap.set(3,800)#寬
    cap.set(4,1200)#高
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 為儲存視訊做準備
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps=30
    out = cv2.VideoWriter('goodby_weijaTest.avi', fourcc,fps,sz)

    startTime = time.time()
    print(type(startTime))
    print(startTime)

    while True:
        # 一幀一幀的獲取影象
        ret,frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 1)
            # 在幀上進行操作
            # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # 開始儲存視訊
            out.write(frame)
            # 顯示結果幀
            end = time.time()
            total = end - startTime
            cv2.imshow("frame", frame)
            print(total)
            cv2.waitKey(1)
            if total > 3:
                break
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    # 釋放攝像頭資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()