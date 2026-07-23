# vllm_geo module

Integrates open-source vision language models served by [vLLM](https://docs.vllm.ai) for geospatial image analysis. vLLM's PagedAttention provides high-throughput self-hosted inference, filling the gap between Ollama (easy but slow) and cloud APIs (fast but costly). This module supports image captioning, visual question answering, and prompt-based object detection with sliding window support for large rasters, in both server mode (`vllm serve <model>`) and in-process mode (`offline=True`).

Suggested models include [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [LLaVA-1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), and [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B).

::: geoai.vllm_geo
