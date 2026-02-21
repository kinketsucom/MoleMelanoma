from logic import MoleMelanoma
from appUI import create_ui

if __name__ == "__main__":
    # インスタンス化
    expert = MoleMelanoma()
    # UIを作成（解析関数を渡す）
    app = create_ui(expert.process)
    # 起動
    app.launch(show_api=False)