import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from neo4j_graphrag.llm import LLMResponse
from neo4j_litellm import LiteLLMInterface, ChatHistory


class TestLiteLLMInterface(unittest.TestCase):
    """测试LiteLLMInterface类"""
    
    def setUp(self):
        """设置测试环境"""
        self.provider = "openai"
        self.model_name = "gpt-3.5-turbo"
        self.base_url = "https://api.openai.com/v1"
        self.api_key = "test-api-key"
        self.llm_interface = LiteLLMInterface(
            provider=self.provider,
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key
        )
    
    def test_initialization(self):
        """测试类初始化"""
        self.assertEqual(self.llm_interface.provider, self.provider)
        self.assertEqual(self.llm_interface.model_name, self.model_name)
        self.assertEqual(self.llm_interface.base_url, self.base_url)
        self.assertEqual(self.llm_interface.api_key, self.api_key)
    
    @patch('neo4j_litellm.completion')
    def test_invoke_basic(self, mock_completion):
        """测试基本同步调用"""
        # 模拟LLM响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, this is a test response"
        mock_completion.return_value = mock_response
        
        # 调用方法
        input_text = "Hello, how are you?"
        response = self.llm_interface.invoke(input_text)
        
        # 验证调用参数
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        self.assertEqual(call_args.kwargs['model'], f"{self.provider}/{self.model_name}")
        self.assertEqual(call_args.kwargs['api_key'], self.api_key)
        self.assertEqual(call_args.kwargs['api_base'], self.base_url)
        self.assertEqual(call_args.kwargs['timeout'], 5)
        
        # 验证消息结构
        messages = call_args.kwargs['messages']
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[0]['content'], input_text)
        
        # 验证响应
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Hello, this is a test response")
    
    @patch('neo4j_litellm.completion')
    def test_invoke_with_message_history(self, mock_completion):
        """测试带消息历史的同步调用"""
        # 模拟LLM响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response with history"
        mock_completion.return_value = mock_response
        
        # 准备消息历史
        message_history = [
            ChatHistory(role="user", content="First message"),
            ChatHistory(role="assistant", content="First response"),
            ChatHistory(role="user", content="Second message")
        ]
        
        # 调用方法
        input_text = "Third message"
        response = self.llm_interface.invoke(input_text, message_history=message_history)
        
        # 验证消息结构包含历史
        messages = mock_completion.call_args.kwargs['messages']
        self.assertEqual(len(messages), 4)  # 3条历史 + 1条新消息
        
        # 验证历史消息顺序
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[0]['content'], 'First message')
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertEqual(messages[1]['content'], 'First response')
        self.assertEqual(messages[2]['role'], 'user')
        self.assertEqual(messages[2]['content'], 'Second message')
        self.assertEqual(messages[3]['role'], 'user')
        self.assertEqual(messages[3]['content'], 'Third message')
    
    @patch('neo4j_litellm.completion')
    def test_invoke_with_system_instruction(self, mock_completion):
        """测试带系统指令的同步调用"""
        # 模拟LLM响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response with system instruction"
        mock_completion.return_value = mock_response
        
        # 调用方法
        input_text = "User question"
        system_instruction = "You are a helpful assistant that speaks like a pirate."
        response = self.llm_interface.invoke(input_text, system_instruction=system_instruction)
        
        # 验证消息结构包含系统指令
        messages = mock_completion.call_args.kwargs['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], system_instruction)
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(messages[1]['content'], input_text)
    
    @patch('neo4j_litellm.completion')
    def test_invoke_with_both_history_and_system(self, mock_completion):
        """测试同时包含消息历史和系统指令的同步调用"""
        # 模拟LLM响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response with both"
        mock_completion.return_value = mock_response
        
        # 准备消息历史
        message_history = [
            ChatHistory(role="user", content="Previous message"),
            ChatHistory(role="assistant", content="Previous response")
        ]
        
        # 调用方法
        input_text = "New message"
        system_instruction = "System instruction"
        response = self.llm_interface.invoke(
            input_text, 
            message_history=message_history,
            system_instruction=system_instruction
        )
        
        # 验证消息结构
        messages = mock_completion.call_args.kwargs['messages']
        self.assertEqual(len(messages), 4)  # 历史 + 系统 + 新消息
        
        # 验证顺序：历史 -> 系统 -> 新消息
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[0]['content'], 'Previous message')
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertEqual(messages[1]['content'], 'Previous response')
        self.assertEqual(messages[2]['role'], 'system')
        self.assertEqual(messages[2]['content'], system_instruction)
        self.assertEqual(messages[3]['role'], 'user')
        self.assertEqual(messages[3]['content'], input_text)
    
    @patch('neo4j_litellm.completion')
    def test_invoke_with_none_parameters(self, mock_completion):
        """测试None参数处理"""
        # 模拟LLM响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response with None params"
        mock_completion.return_value = mock_response
        
        # 调用方法，所有可选参数为None
        input_text = "Test message"
        response = self.llm_interface.invoke(
            input_text, 
            message_history=None,
            system_instruction=None
        )
        
        # 验证只有用户消息
        messages = mock_completion.call_args.kwargs['messages']
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[0]['content'], input_text)


class TestLiteLLMInterfaceAsync(unittest.IsolatedAsyncioTestCase):
    """测试LiteLLMInterface的异步方法"""
    
    def setUp(self):
        """设置测试环境"""
        self.provider = "openai"
        self.model_name = "gpt-3.5-turbo"
        self.base_url = "https://api.openai.com/v1"
        self.api_key = "test-api-key"
        self.llm_interface = LiteLLMInterface(
            provider=self.provider,
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key
        )
    
    @patch('neo4j_litellm.acompletion')
    async def test_ainvoke_basic(self, mock_acompletion):
        """测试基本异步调用"""
        # 模拟异步LLM响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Async response"
        mock_acompletion.return_value = mock_response
        
        # 调用异步方法
        input_text = "Async test message"
        response = await self.llm_interface.ainvoke(input_text)
        
        # 验证调用参数
        mock_acompletion.assert_called_once()
        call_args = mock_acompletion.call_args
        self.assertEqual(call_args.kwargs['model'], f"{self.provider}/{self.model_name}")
        self.assertEqual(call_args.kwargs['api_key'], self.api_key)
        self.assertEqual(call_args.kwargs['api_base'], self.base_url)
        self.assertEqual(call_args.kwargs['timeout'], 5)
        
        # 验证响应
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Async response")
    
    @patch('neo4j_litellm.acompletion')
    async def test_ainvoke_with_history_and_system(self, mock_acompletion):
        """测试带消息历史和系统指令的异步调用"""
        # 模拟异步LLM响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Async response with both"
        mock_acompletion.return_value = mock_response
        
        # 准备消息历史
        message_history = [
            ChatHistory(role="user", content="Async history message"),
            ChatHistory(role="assistant", content="Async history response")
        ]
        
        # 调用异步方法
        input_text = "Async new message"
        system_instruction = "Async system instruction"
        response = await self.llm_interface.ainvoke(
            input_text,
            message_history=message_history,
            system_instruction=system_instruction
        )
        
        # 验证消息结构
        messages = mock_acompletion.call_args.kwargs['messages']
        self.assertEqual(len(messages), 4)
        
        # 验证顺序正确
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[0]['content'], 'Async history message')
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertEqual(messages[1]['content'], 'Async history response')
        self.assertEqual(messages[2]['role'], 'system')
        self.assertEqual(messages[2]['content'], system_instruction)
        self.assertEqual(messages[3]['role'], 'user')
        self.assertEqual(messages[3]['content'], input_text)


class TestChatHistory(unittest.TestCase):
    """测试ChatHistory数据结构"""
    
    def test_chat_history_creation(self):
        """测试ChatHistory创建"""
        chat_history = ChatHistory(role="user", content="Test message")
        self.assertEqual(chat_history['role'], "user")
        self.assertEqual(chat_history['content'], "Test message")
    
    def test_chat_history_types(self):
        """测试ChatHistory类型验证"""
        # 应该能够创建不同角色的消息
        system_msg = ChatHistory(role="system", content="System message")
        assistant_msg = ChatHistory(role="assistant", content="Assistant message")
        user_msg = ChatHistory(role="user", content="User message")
        
        self.assertEqual(system_msg['role'], "system")
        self.assertEqual(assistant_msg['role'], "assistant")
        self.assertEqual(user_msg['role'], "user")


if __name__ == '__main__':
    unittest.main()