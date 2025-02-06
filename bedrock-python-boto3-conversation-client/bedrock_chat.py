import boto3
import json
import sys
from datetime import datetime

# 指令說明字典
COMMAND_HELP = {
    "help": {
        "指令": "help",
        "說明": "顯示所有可用指令的說明",
        "使用方式": "直接輸入 'help' 或 'help 指令名稱'"
    },
    "quit": {
        "指令": "quit",
        "說明": "結束對話並退出程式",
        "使用方式": "直接輸入 'quit' 即可"
    },
    "clear": {
        "指令": "clear",
        "說明": "清除當前的對話歷史記錄",
        "使用方式": "直接輸入 'clear' 即可"
    },
    "save": {
        "指令": "save",
        "說明": "將當前對話歷史儲存到檔案",
        "使用方式": "直接輸入 'save' 即可，將儲存為 chat_history.json"
    },
    "load": {
        "指令": "load",
        "說明": "從檔案載入之前儲存的對話歷史",
        "使用方式": "直接輸入 'load' 即可，將從 chat_history.json 載入"
    },
    "role": {
        "指令": "role",
        "說明": "查看並切換 AI 助手的角色",
        "使用方式": "1. 輸入 'role' 查看可用角色\n" +
                   "2. 根據提示輸入想要切換的角色名稱"
    }
}

# Role Prompt 模組區域
ROLE_PROMPTS = {
    "default": """You are Claude, an AI assistant created by Anthropic. 
                 You are helpful, harmless, and honest.""",
    
    "translator": """You are a professional translator. 
                    Always maintain the original meaning while providing natural translations. 
                    When translating, consider cultural context and local expressions.
                    Please respond in the target language directly.""",
    
    "programmer": """You are an experienced software developer.
                    Provide clear, well-documented code examples.
                    Focus on best practices and explain your reasoning.
                    Always include comments in Traditional Chinese.""",
    
    "teacher": """You are a patient and knowledgeable teacher.
                 Break down complex concepts into simple terms.
                 Use examples and analogies to explain difficult topics.
                 Please respond in Traditional Chinese.""",
    
    "business_analyst": """You are a business analysis expert.
                         Provide detailed market insights and data-driven analysis.
                         Focus on practical business recommendations.
                         Please respond in Traditional Chinese.""",
    
    "aws_sa": """你是一個 AWS Solutions Architect.
                         你會用台灣中文回答關於AWS的所有技術、架構問題，但遇到專有名詞你會使用原來的英文不會進行翻譯。
                         說明過程要有條有理"""
}

class BedrockChat:
    def __init__(self, role_type="default"):
        """
        初始化 BedrockChat 類別
        Args:
            role_type (str): 選擇預設的角色類型
        """
        # 初始化 AWS Bedrock 客戶端
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'  # AWS 區域設定
        )
        self.conversation_history = []
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        self.current_role = role_type
        self.available_commands = COMMAND_HELP

    def show_help(self, command=None):
        """
        顯示指令說明
        Args:
            command (str, optional): 特定指令名稱。若未指定則顯示所有指令說明
        """
        if command and command in self.available_commands:
            cmd_info = self.available_commands[command]
            print(f"\n【{cmd_info['指令']}】指令說明：")
            print(f"說明：{cmd_info['說明']}")
            print(f"使用方式：{cmd_info['使用方式']}")
        else:
            print("\n=== 可用指令說明 ===")
            for cmd, info in self.available_commands.items():
                print(f"\n【{info['指令']}】")
                print(f"說明：{info['說明']}")
                print(f"使用方式：{info['使用方式']}")
            print("\n==================")

    def show_current_role(self):
        """顯示當前角色設定"""
        print(f"\n當前角色：{self.current_role}")
        print(f"角色設定：{ROLE_PROMPTS[self.current_role][:100]}...")

    def set_role(self, role_type):
        """
        設定 AI 助手的角色
        Args:
            role_type (str): 要設定的角色類型
        """
        if role_type in ROLE_PROMPTS:
            self.current_role = role_type
            return True
        return False

    def add_to_history(self, role, content):
        """
        將對話內容添加到歷史記錄中
        Args:
            role (str): 發言者角色 (user/assistant)
            content (str): 對話內容
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_response(self, user_input):
        """
        獲取 AI 模型的回應
        Args:
            user_input (str): 使用者輸入的內容
        Returns:
            str: AI 的回應內容
        """
        try:
            self.add_to_history("user", user_input)

            messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"]
                } for msg in self.conversation_history
            ]

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 250,
                "system": ROLE_PROMPTS[self.current_role],
            })

            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response.get('body').read())
            assistant_response = response_body['content'][0]['text']

            self.add_to_history("assistant", assistant_response)

            return assistant_response

        except Exception as e:
            print(f"錯誤: {str(e)}")
            return None

    def clear_history(self):
        """清除所有對話歷史"""
        self.conversation_history = []

    def save_conversation(self, filename):
        """
        儲存對話歷史到檔案
        Args:
            filename (str): 要儲存的檔案名稱
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)

    def load_conversation(self, filename):
        """
        從檔案載入對話歷史
        Args:
            filename (str): 要載入的檔案名稱
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print("對話歷史已載入")
        except FileNotFoundError:
            print("找不到歷史對話檔案")

def main():
    """主程式執行函數"""
    chat = BedrockChat()
    
    print("\n=== 歡迎使用 AWS Bedrock Chat (Claude 3 Sonnet) ===")
    print("輸入 'help' 查看所有可用指令")
    print("================================================")

    while True:
        try:
            user_input = input("\n你: ").strip()

            if user_input.lower() == 'help':
                chat.show_help()
                continue
            elif user_input.lower().startswith('help '):
                specific_command = user_input.lower().split()[1]
                chat.show_help(specific_command)
                continue
            elif user_input.lower() == 'quit':
                print("感謝使用！再見！")
                break
            elif user_input.lower() == 'clear':
                chat.clear_history()
                print("對話歷史已清除！")
                continue
            elif user_input.lower() == 'save':
                chat.save_conversation('chat_history.json')
                print("對話歷史已儲存")
                continue
            elif user_input.lower() == 'load':
                chat.load_conversation('chat_history.json')
                continue
            elif user_input.lower() == 'role':
                print("\n可用角色：")
                for role in ROLE_PROMPTS.keys():
                    print(f"- {role}")
                chat.show_current_role()
                new_role = input("請選擇角色 (直接按 Enter 維持當前角色): ").strip()
                if new_role:
                    if chat.set_role(new_role):
                        print(f"已切換至 {new_role} 角色")
                    else:
                        print("無效的角色類型")
                continue
            elif not user_input:
                continue

            response = chat.get_response(user_input)
            
            if response:
                print("\nClaude:", response)
            else:
                print("\n抱歉，出現錯誤，請重試。")

        except KeyboardInterrupt:
            print("\n程式已中止")
            break
        except Exception as e:
            print(f"\n發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()