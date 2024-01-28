import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt


# 必要な関数の定義（元のコードから）
def topbottom(img):
    # ...（元のコードのtopbottom関数の内容）
    img = (img > 128) * 255    
  
    # 配列の行数と列数を取得
    rows, cols = img.shape
    
    # 右上の端の座標を見つける。行は昇順。列は逆から。白（255）を見つけた場所の関数。
    #1回目の上端検出
    for i in range(rows):
        for j in range(cols - 1, -1, -1):
            if img[i, j] == 255:
                img_top = (i, j)                
                break
        if 'img_top' in locals():
            break
    #2回目の上端検出
    img_top_list = []
    img_y = img_top[0] +10
    for j in range(cols-1,-1,-1):
        if img[img_y,j] == 255:
            img_top_list.append(j)
            break
    for j in range(-1,cols-1,1):
        if img[img_y,j] == 255:
            img_top_list.append(j)
            break
    img_top = [img_y,np.average(img_top_list)]

    # 左下の端の座標を見つける。行は逆から。列は昇順
    for i in range(rows - 1, -1, -1):
        for j in range(cols):
            if img[i, j] == 255:
                img_bottom = (i, j)
                break
        if 'img_bottom' in locals():
            break

    # print("右上の端の座標:",img_top)
    # print("左下の端の座標:", img_bottom)
    return img_top, img_bottom

def image_press(or_im):
    # ...（元のコードのimage_press関数の内容）
     #余白の追加
    im = or_im

    def expand2square(pil_img, background_color):
      width, height = pil_img.size
      if width == height:
          return pil_img
      elif width > height:
          result = Image.new(pil_img.mode, (width, width), background_color)
          result.paste(pil_img, (0, (width - height) // 2))
          return result
      else:
          result = Image.new(pil_img.mode, (height, height), background_color)
          result.paste(pil_img, ((height - width) // 2, 0))
          return result

    im_new = expand2square(im, (255, 255, 255))
    im_new = im_new.resize((416,416))
    return im_new

# Streamlitウィジェットの設定
st.title("画像アップロードまたはカメラで撮影")

# 画像のアップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# カメラでの画像撮影
camera_image = st.camera_input("またはカメラで画像を撮影してください")

# 画像の読み込み
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif camera_image is not None:
    image = Image.open(camera_image)

# 画像の前処理と物体検出の実行
if 'image' in locals():
    # 画像を処理可能な形式に変換
    org_img = image_press(image)
    img_size=300
    # 物体検出の実行
    if st.button('物体検出', key='my_button'):
        # モデルの読み込み
        model = YOLO("e_meter_segadd2.pt")

        # 画像の予測
        results = model.predict(org_img, imgsz=416, conf=0.5, classes=0)
        # ...（物体検出後の処理、元のコードの該当部分を使用）
        #エラー対策（検出ゼロと，検出４未満なのに白検出ゼロのとき）
        if results[0].masks==None:
            st.error("配線の検出に失敗しました。目視で確認してください",icon="🚨")
            st.image(org_img,width=img_size)
            sys.exit("配線の検出に失敗しました。目視で確認してください")
        if len(results[0].masks) < 3:
            for r in results:
                im_array = r.plot(boxes=False)
                im = Image.fromarray(im_array[..., ::-1])
            st.error("配線の検出に失敗しました。目視で確認してください",icon="🚨")
            st.image(im,width=img_size)
            sys.exit("配線の検出に失敗しました。目視で確認してください")
    
    
    # 処理結果を格納するための辞書
        processed_data = {
         'images': [],
          'coordinates': [],
            'classification': ''
        }
    
    # 各マスク画像の処理
        for i, r in enumerate(results[0].masks):
            # マスク画像情報を(0,1)から(0,255)に変換し、int型にキャスト
            mask_img = r.data[0].cpu().numpy() * 255
            mask_img = mask_img.astype(int)
        
            # 上下の端座標を求める
            top, bottom = topbottom(mask_img)
            
             # 画像データと座標データを辞書に追加
            processed_data['images'].append(mask_img)
            processed_data['coordinates'].append((top, bottom))
    
    
    # 座標のリストを取得。タプルからリストに変換
        coordinates_list = processed_data['coordinates']
        connect_list = np.array(coordinates_list)
    
    #コネクタの上端と下端の座標ごとのリストに分ける
        bottom_list=[]
        top_list= []
        for r in connect_list:
            top_list.append(r[0])
            bottom_list.append(r[1])
    
        bottom_list = np.array(bottom_list)
        top_list = np.array(top_list)

        #----------検出３本の場合--------------
    #抽出本数が３本の場合，白線の情報を取得する
        if len(top_list)==3:
        
        #上端データの作成
        #赤Lと黒K用の上端データを作る
            r=[]
            b=[]
        #赤黒３線分の情報をxとyの各リストに分ける（x1,x2,x3になる）
            y0, x0 = zip(*sorted(top_list, key=lambda x: x[1]))
    
        #x1-x2 x2-x3間の距離を取得
            xl1 = x0[1] - x0[0]
            xl2 = x0[2] - x0[1]
    
        #xl1とxl2で少ない方がwidth。xl1が少ないなら黒が欠損，xl2なら赤が欠損  
        #黒欠損
            if xl1 < xl2:
                width = xl1
                B1 = abs(xl2-width*3)
                B2 = abs(xl2-width*2)
                #少ない方が欠損部分。そこに合わせて補完する
                if B1<B2:
                    topx_dummy = x0[2] - width
                else:
                    topx_dummy = x0[2] + width
        #赤欠損
            else:
                width = xl2
                R1 = abs(xl1-width*2)
                R2 = abs(xl1-width*3)
                #少ない方が欠損部分。そこに合わせて補完する
                if R1<R2:
                    topx_dummy = x0[0] - width
                else:
                    topx_dummy = x0[0] + width 
            
            #高さyの平均
            y_avr=int(sum(y0)/len(y0))
            
            #仮定xと平均y座標をtop_listに追加する。
            top_list=np.append(top_list,[(y_avr,topx_dummy)],axis=0)
    
            #先にソートして，R1+width*2して中心（仮）を取得
            sorted_comptop = np.array(sorted(top_list, key=lambda x: x[1]))   
            connect_center = sorted_comptop[0][1]+width*2
    
            #下端データの作成
            #赤黒３線分の下端情報を分けてソート
            k=[]
            l=[]
            y0, x0 = zip(*sorted(bottom_list, key=lambda x: x[1]))
            y_avr=int(sum(y0)/len(y0))
    
            #中心を境にKLに振り分け
            for line in x0:
                if line < connect_center:
                    k=np.append(k,line)
                else:
                    l=np.append(l,line)
      
        #１つしかない方がlenで1。少ない方にダミー数値追加して補完（円描画用にもう一つの下端±5くらい）
            if len(k) == 1:
                btmx_dummy=k[0]-5
            if len(l) == 1:
                btmx_dummy=l[0]+5
        bottom_list=np.append(bottom_list,[(y_avr,btmx_dummy)],axis=0)
            
        #----------以下共通----------
        # ラムダ式を使って上端の2番目をキーとしてソート＝上端の横座標順のリストで並べる。下端も上端の順番に追従
        sorted_top = np.array(sorted(top_list, key=lambda x: x[1]))
        sorted_bottom = np.array([tup for _, tup in sorted(zip(top_list, bottom_list), key=lambda x: x[0][1])])
        
        # 分類結果の計算
        center = sum([coords[1] for coords in sorted_bottom]) / 4
        
        #0,2行目（K側）と1,3行目（L側）と中央値を比較して判別
        if np.all(sorted_bottom[::2,1] < center) and np.all(sorted_bottom[1::2,1] > center):
            st.success("左電源の正結線の可能性が高いです",icon = "✅")
        elif np.all(sorted_bottom[::2,1] > center) and np.all(sorted_bottom[1::2,1] < center):
            st.success("右電源の正結線の可能性が高いです",icon = "✅")
        else:
            st.warning("誤配線の可能性があります。目視で確認してください",icon="⚠")
        
        # st.write(processed_data['classification'])
        
        #画像の形式に整える
        for r in results:
            im_array = r.plot(boxes=False)
            im = Image.fromarray(im_array[..., ::-1])
        
        # 新しい画像の作成
        draw = ImageDraw.Draw(im)
        
        # topとbottomのそれぞれの座標に基づいて円を描画
        for i, (y, x) in enumerate(sorted_top):
            x1, y1, x2, y2 = x - 5, y - 5, x + 5, y + 5
            fill_color = (255, 0, 255) if i < 2 else (0, 0, 255)
            draw.ellipse((x1, y1, x2, y2), fill=fill_color)
            
        
        for i, (y, x) in enumerate(sorted_bottom):
            x1, y1, x2, y2 = x - 5, y - 5, x + 5, y + 5
            fill_color = (255, 0, 255) if i < 2 else (0, 0, 255)
            draw.ellipse((x1, y1, x2, y2), fill=fill_color)
        
        # 画像の表示
        st.image(im,width=img_size)
        # im.save("./aa.jpg")
            
    st.write("検出前")            
    st.image(org_img,width=img_size)

            
    # 結果の表示
    # ...（結果表示のためのコード、元のコードの該当部分を使用）
