import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class MoleMelanoma:
    def __init__(self, model_type="vit_b", checkpoint_path="weights/sam_vit_b_01ec64.pth"):#ここから拝借https://github.com/bowang-lab/MedSAM?tab=readme-ov-file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)

    def _get_asymmetry(self, mask_uint8):
        #Asymmetry
        m = cv2.moments(mask_uint8)
        if m["m00"] == 0: return 0.0
        flipped_v = cv2.flip(mask_uint8, 1)
        intersection = np.logical_and(mask_uint8, flipped_v).sum()
        union = np.logical_or(mask_uint8, flipped_v).sum()
        return round(1 - (intersection / union), 3)

    def _get_border(self, mask_uint8):
        #Border
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0.0
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: return 0.0
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return round(1 - circularity, 3)

    def _get_color_var(self, image_bgr, mask):
        #Color
        pixels = image_bgr[mask]
        if len(pixels) == 0: return 0.0
        
        hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        std_devs = np.std(hsv_pixels, axis=0)
        return round(min(np.mean(std_devs) / 50.0, 1.0), 3)

    #A,B,Cについて
    def analyze_lesion(self, image_bgr, mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        a = self._get_asymmetry(mask_uint8)
        b = self._get_border(mask_uint8)
        c = self._get_color_var(image_bgr, mask)
        
        return a, b, c

    def process(self, image_pil, d_input, e_input):
        """Gradioから呼ばれるメイン処理"""
        # 画像未入力チェック
        if image_pil is None:
            return None, 0.0, 0.0, 0.0, "画像をアップロードしてください。"

        # 画像の準備
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        h, w, _ = image_bgr.shape
        
        # SAM によるセグメンテーション
        self.predictor.set_image(image_bgr)
        input_point = np.array([[w // 2, h // 2]]) # 画像中心をターゲットに
        input_label = np.array([1])
        
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point, 
            point_labels=input_label, 
            multimask_output=True
        )
        mask = masks[np.argmax(scores)] # 最も確信度の高いマスクを選択
        
        # ABC 解析 (分割したメソッドを呼び出す)
        a, b, c = self.analyze_lesion(image_bgr, mask)
        
        # D, E スコアの加算
        d_score = 0.2 if d_input == "6mm以上" else 0.0
        e_score = 0.3 if e_input else 0.0
        
        # 総合リスク計算
        total_risk = round(min((a + b + c) / 3.0 + d_score + e_score, 1.0), 3)
        
        # 判定メッセージの生成はしているが、ここはテキトーなサンプル
        if total_risk > 0.7:
            judgment = f"【要専門医受診】悪性黒色腫の可能性が否定できません。 (Total: {total_risk})"
        elif total_risk > 0.4:
            judgment = f"【経過観察推奨】不規則な特徴が見られます。皮膚科受診を検討してください。 (Total: {total_risk})"
        else:
            judgment = f"【低リスク】現時点では良性の特徴が強いです。 (Total: {total_risk})"

        # 結果画像の作成 (オーバーレイ表示)
        overlay = image_bgr.copy()
        # 青色で透過塗りつぶし
        overlay[mask] = (overlay[mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8) 
        
        # 境界線の描画
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(overlay, [cnt], -1, (255, 255, 255), 2)

        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), a, b, c, judgment