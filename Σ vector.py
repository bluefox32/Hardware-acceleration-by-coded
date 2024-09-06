import cv2
import numpy as np
import matplotlib.pyplot as plt

# 動画からフレームを取得し、画像処理を行う
def process_video(video_path, block_size):
    cap = cv2.VideoCapture(video_path)
    all_vectors = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # グレースケールに変換
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 画像データをブロックに分割
        blocks = split_into_blocks(gray_frame, block_size)
        
        # ブロックの総和と平均を計算してベクトル化
        vectors = compute_block_vectors(blocks)
        all_vectors.append(vectors)
    
    cap.release()
    
    # 結果を可視化（例: 最後のフレームのベクトルを表示）
    if all_vectors:
        plot_vectors(all_vectors[-1])

if __name__ == "__main__":
    # 動画ファイルのパス
    video_path = 'sample_video.mp4'
    
    # ブロックサイズ（8x8など）を指定
    block_size = 8
    
    # 動画を処理
    process_video(video_path, block_size)