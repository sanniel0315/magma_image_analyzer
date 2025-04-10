import torch
from PIL import Image
import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")

class MagmaAnalyzer:
    def __init__(self, use_gpu=True):
        # 检查GPU是否可用
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"初始化Magma-8B模型 (设备: {self.device})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Magma-8B", 
            trust_remote_code=True, 
            torch_dtype=self.dtype
        )
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Magma-8B", 
            trust_remote_code=True
        )
        
        self.model = self.model.to(self.device)
        print("模型初始化完成")
    
    def analyze_image(self, image, prompt_text="请用繁体中文描述這張圖片", max_tokens=512):
        """分析图像并返回结果"""
        if image is None:
            return "请上傳圖片"
        
        # 确保图像是RGB模式
        image = image.convert("RGB")
        
        # 使用繁体中文的对话格式
        convs = [
            {"role": "system", "content": "你是一个能看、能说的智能助手。请務必只使用繁体中文回答所有問题。"},
            {"role": "user", "content": f"<image_start><image><image_end>\n{prompt_text}，務必使用繁体中文回答。"},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        
        # 处理输入
        inputs = self.processor(images=[image], texts=prompt, return_tensors="pt")
        
        # 修正数据类型
        if 'pixel_values' in inputs:
        # 确保维度正确
            if inputs['pixel_values'].dim() == 4:  # 如果维度是[batch, channels, height, width]
                inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)  # 添加一个维度
        
        if 'image_sizes' in inputs:
        # 确保是整数类型
            inputs['image_sizes'] = inputs['image_sizes'].long()
            
        # 确保维度正确
        if inputs['image_sizes'].dim() == 2:  # 如果维度是[batch, 2]
            inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)  # 变成[1, batch, 2]
    
        # 修正数据类型
        for key in ['input_ids', 'attention_mask']:
            if key in inputs:
                inputs[key] = inputs[key].long()
    
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成参数
        generation_args = { 
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "use_cache": True,
            "num_beams": 1,
        }
        
        # 生成回答
        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, **generation_args)
        
        # 解码输出
        generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        response = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()
        
        return response

# 创建全局的分析器实例
analyzer = None

def analyze_image_wrapper(image, prompt, use_gpu, max_tokens):
    """包装函数，处理图像分析的主要逻辑"""
    global analyzer
    
    # 延迟加载模型
    if analyzer is None or analyzer.device != ("cuda" if use_gpu and torch.cuda.is_available() else "cpu"):
        try:
            analyzer = MagmaAnalyzer(use_gpu=use_gpu)
        except Exception as e:
            return f"加载模型失败: {str(e)}"
    
    try:
        result = analyzer.analyze_image(image, prompt, int(max_tokens))
        return result
    except Exception as e:
        return f"分析图像时出错: {str(e)}"

# 创建Gradio界面
with gr.Blocks(title="Magma-8B图像分析") as demo:
    gr.Markdown("# Magma-8B 图像分析工具")
    gr.Markdown("上传图片并获取繁体中文描述")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 输入区域
            input_image = gr.Image(type="pil", label="上传图片")
            prompt = gr.Textbox(label="提示文本", value="请用繁体中文详细描述这张图片的内容", lines=2)
            
            with gr.Row():
                # 参数设置
                use_gpu = gr.Checkbox(label="使用GPU", value=True)
                max_tokens = gr.Slider(minimum=100, maximum=1024, value=512, step=32, label="最大生成长度")
            
            analyze_btn = gr.Button("分析图片", variant="primary")
        
        with gr.Column(scale=1):
            # 输出区域
            output_text = gr.Textbox(label="分析结果", lines=15)
    
    # 绑定分析按钮事件
    analyze_btn.click(
        fn=analyze_image_wrapper,
        inputs=[input_image, prompt, use_gpu, max_tokens],
        outputs=output_text
    )
    
    # 使用说明
    gr.Markdown("""
    ## 使用说明
    1. 上传图片或使用示例图片
    2. 输入提示文本（预设为"请用繁体中文详细描述这张图片的内容"）
    3. 选择是否使用GPU（如果可用）
    4. 点击"分析图片"按钮获取结果
    
    ## 关于模型
    本应用使用Microsoft的Magma-8B多模态模型进行图像分析。模型可以理解图像内容并生成详细的文本描述。
    """)

# 启动Gradio应用
if __name__ == "__main__":
    demo.launch()
else:
    # 这是重要的部分，用于Spaces部署
    app = demo