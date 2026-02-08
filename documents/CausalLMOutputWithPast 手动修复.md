探索报告：CausalLMOutputWithPast 手动修复问题
1. CausalLMOutputWithPast 相关代码位置
主要位置：third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py
关键修复代码（第 460-466 行）：
# Handle both BaseModelOutputWithPast and CausalLMOutputWithPastif hasattr(suffix_output, 'last_hidden_state'):    suffix_output = suffix_output.last_hidden_stateelif hasattr(suffix_output, 'hidden_states') and suffix_output.hidden_states:    suffix_output = suffix_output.hidden_states[-1]else:    raise AttributeError("Cannot extract hidden states from model output")
上下文（第 446-468 行）：
当 inputs_embeds[0] 为 None 时（仅处理 expert model）
处理 PEFT-wrapped models：需要从 gemma_expert.model 中提取 GemmaModel
调用 expert_model 的 forward，设置 output_hidden_states=True
手动处理不同的输出类型
2. last_hidden_state 的使用情况
PI05 中的使用：
第 444 行：prefix_output = prefix_output.last_hidden_state（直接访问）
第 461-462 行：通过 hasattr 检查后访问（修复代码）
第 463-464 行：fallback 到 hidden_states[-1]
其他文件中的使用：
third_party/lerobot/src/lerobot/policies/pi0/modeling_pi0.py:436：直接访问 suffix_output.last_hidden_state（无修复）
third_party/lerobot/src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py:304-305：类似检查模式
third_party/lerobot/src/lerobot/policies/sac/reward_model/modeling_classifier.py:79：直接访问
3. PI05 从 VLM 提取特征的方式
特征提取流程：
前缀（prefix）处理（第 435-445 行）：
使用 paligemma.language_model.forward()
直接访问 prefix_output.last_hidden_state
提取 past_key_values
后缀（suffix）处理（第 446-468 行）：
处理 PEFT-wrapped models：从 gemma_expert.model 提取 GemmaModel
调用 expert_model.forward()，设置 output_hidden_states=True
使用修复代码处理不同的输出类型
同时处理两者（第 469-520 行）：
使用 compute_layer_complete 函数
处理 PEFT-wrapped models（第 214-217 行）
4. PEFT/LoRA 相关的输出结构变化
PEFT 包装导致的结构变化：
模型结构变化（多处处理）：
   # 第 214-217 行，第 447-450 行，第 470-473 行   expert_model = self.gemma_expert.model   if hasattr(expert_model, 'model') and hasattr(expert_model.model, 'layers'):       expert_model = expert_model.model
PEFT 包装后：gemma_expert.model 可能是 GemmaForCausalLM
需要进一步访问 .model 才能得到 GemmaModel
输出类型变化：
未包装：返回 BaseModelOutputWithPast（有 last_hidden_state）
PEFT 包装后：可能返回 CausalLMOutputWithPast（有 logits，可能没有 last_hidden_state）
或者返回包含 hidden_states 列表的输出
修复策略：
优先检查 last_hidden_state
否则使用 hidden_states[-1]
如果都没有则抛出异常
相关文件：
kuavo_train/wrapper/policy/pi05/PI05ModelWrapper.py:58-98：LoRA 应用逻辑
third_party/vlash/vlash/lora/apply.py:118-200：PEFT LoRA 应用
third_party/lerobot/my_lerobot_changes.patch:89,155：PEFT 处理相关的补丁
5. 对比：PI0 vs PI05
PI0（无修复）：
third_party/lerobot/src/lerobot/policies/pi0/modeling_pi0.py:436
直接访问：suffix_output = suffix_output.last_hidden_state
没有处理 PEFT 包装后的输出类型变化
PI05（有修复）：
添加了输出类型兼容性检查
处理 PEFT-wrapped models 的结构变化
支持多种输出格式
总结
手动修复的原因：
PEFT/LoRA 包装后，模型输出类型可能从 BaseModelOutputWithPast 变为 CausalLMOutputWithPast
CausalLMOutputWithPast 可能没有 last_hidden_state，但有 hidden_states 列表
需要兼容不同的输出格式以确保特征提取的稳定性
修复位置：
主要修复：third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py:460-466
PEFT 模型结构处理：第 214-217 行，第 447-450 行，第 470-473 行
相关文件路径：
/home/yly/ICRA-kuavo/kuavo_data_challenge/third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py
/home/yly/ICRA-kuavo/kuavo_data_challenge/third_party/lerobot/src/lerobot/policies/pi0/modeling_pi0.py
/home/yly/ICRA-kuavo/kuavo_data_challenge/third_party/lerobot/my_lerobot_changes.patch

CausalLMOutputWithPast 手动修复验证报告
1. 修复代码详情
位置：third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py 第 446-466 行
修复代码：
elif inputs_embeds[0] is None:    # Handle PEFT-wrapped models: gemma_expert.model may return CausalLM, need .model for GemmaModel    expert_model = self.gemma_expert.model    if hasattr(expert_model, 'model') and hasattr(expert_model.model, 'layers'):        expert_model = expert_model.model    suffix_output = expert_model.forward(        inputs_embeds=inputs_embeds[1],        attention_mask=attention_mask,        position_ids=position_ids,        past_key_values=past_key_values,        use_cache=use_cache,        output_hidden_states=True,  # 关键：显式请求 hidden_states        adarms_cond=adarms_cond[1] if adarms_cond is not None else None,    )    # Handle both BaseModelOutputWithPast and CausalLMOutputWithPast    if hasattr(suffix_output, 'last_hidden_state'):        suffix_output = suffix_output.last_hidden_state    elif hasattr(suffix_output, 'hidden_states') and suffix_output.hidden_states:        suffix_output = suffix_output.hidden_states[-1]    else:        raise AttributeError("Cannot extract hidden states from model output")
2. 对比 PI0 的实现
PI0 实现（modeling_pi0.py 第 427-436 行）：
elif inputs_embeds[0] is None:    suffix_output = self.gemma_expert.model.forward(        inputs_embeds=inputs_embeds[1],        attention_mask=attention_mask,        position_ids=position_ids,        past_key_values=past_key_values,        use_cache=use_cache,        adarms_cond=adarms_cond[1] if adarms_cond is not None else None,    )    suffix_output = suffix_output.last_hidden_state  # 直接访问，无容错
差异：
PI0：直接访问 last_hidden_state，未处理 PEFT 包装后的输出类型变化
PI05：添加了兼容性检查，支持多种输出格式
3. hidden_states 的正确性验证
hidden_states[-1] 与 last_hidden_state 的关系：
根据 transformers 库的文档和实现：
last_hidden_state：最后一层的输出，形状为 (batch_size, sequence_length, hidden_size)
hidden_states：当 output_hidden_states=True 时返回的元组，包含所有层的 hidden states（包括 embedding 层）
hidden_states[-1]：通常是最后一层的输出
等价性：
在大多数情况下，hidden_states[-1] 与 last_hidden_state 等价
某些模型可能在最后应用额外的归一化或后处理，导致细微差异
对于 Gemma 模型，两者应等价
修复代码的正确性：
优先使用 last_hidden_state（如果存在）
否则使用 hidden_states[-1] 作为后备
这是合理的容错策略
4. PEFT 包装后的模型输出结构
PEFT 包装的影响：
未包装：GemmaModel.forward() 返回 BaseModelOutputWithPast（有 last_hidden_state）
PEFT 包装后：GemmaForCausalLM.forward() 可能返回 CausalLMOutputWithPast（有 logits，可能没有 last_hidden_state）
修复代码的处理：
第 447-450 行：处理 PEFT 包装后的模型结构，从 GemmaForCausalLM 提取 GemmaModel
第 457 行：显式设置 output_hidden_states=True，确保返回 hidden_states
第 461-464 行：兼容性检查，优先使用 last_hidden_state，否则使用 hidden_states[-1]
5. 维度问题检查
维度分析：
last_hidden_state 形状：(batch_size, sequence_length, hidden_size)
hidden_states[-1] 形状：(batch_size, sequence_length, hidden_size)
两者形状应一致，无需额外 squeeze 或 reshape
潜在问题：
hidden_states 可能是 None：代码已检查 suffix_output.hidden_states
hidden_states 可能是空元组：代码已检查 suffix_output.hidden_states
维度不匹配：在 Gemma 模型中不应出现
6. 修复的正确性评估
优点：
兼容性：同时支持 BaseModelOutputWithPast 和 CausalLMOutputWithPast
容错性：有明确的错误处理
优先级：优先使用 last_hidden_state（更直接）
PEFT 支持：正确处理 PEFT 包装后的模型结构
潜在问题：
hidden_states[-1] 与 last_hidden_state 的细微差异：对于 Gemma 模型，影响可忽略
PI0 未修复：PI0 在相同场景下可能出错，建议同步修复
建议改进：
# 可以考虑添加日志记录，便于调试if hasattr(suffix_output, 'last_hidden_state'):    suffix_output = suffix_output.last_hidden_stateelif hasattr(suffix_output, 'hidden_states') and suffix_output.hidden_states:    suffix_output = suffix_output.hidden_states[-1]    # 可选：添加警告日志    # logging.warning("Using hidden_states[-1] instead of last_hidden_state")else:    raise AttributeError("Cannot extract hidden states from model output")
7. 总结
修复是正确的：
正确处理了 PEFT 包装后的输出结构变化
兼容多种输出格式
有适当的错误处理
hidden_states[-1] 与 last_hidden_state 在 Gemma 模型中等价
无需额外的维度处理
建议：
考虑在 PI0 中应用相同的修复
可选：添加日志记录以便调试
验证：在实际使用 PEFT 包装的模型时测试该修复
修复代码位置：
/home/yly/ICRA-kuavo/kuavo_data_challenge/third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py:446-466
根据三个探索任务的结果，我现在有了比较全面的分析。让我总结一下关键发现：
关键发现
1. Stats加载问题（最可能的原因）
在 sim_auto_test.py:371 中，make_pre_post_processors(None, ...) 没有传入正确的stats
如果stats没有正确加载，create_stats_buffers 会将统计量初始化为 torch.inf
使用 torch.inf 进行unnormalize会产生 inf 或 nan 值
2. CausalLMOutput修复是正确的
修复代码看起来没问题
hidden_states[-1] 与 last_hidden_state 在Gemma模型中等价
3. Action Chunking逻辑