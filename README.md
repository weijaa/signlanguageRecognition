# 手語辨識
-  11種手語
   1. 吃 (eat)
   2. 再見 (goodbye)
   3. 你好 (hello)
   4. 幫幫我 (help me)
   5. 早安 (good morning)
   6. 晚安 (good night)
   7. 請 (please)
   8. 睡覺 (sleep)
   9. 抱歉 (sorry)
   10. 謝謝 (thanks)
   11. 不客氣 (welcome)
- 440部手語影片

## 研究步驟
   1. 利用openpose取出骨架資訊
   2. 計算每個關節點彼此的距離關係，變成55*55的距離矩陣
   3. 丟進自行訓練的cnn+lstm模型
   4. 得到結果

