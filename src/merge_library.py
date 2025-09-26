from utils import load_data
import os
from utils import get_model
from peft import PeftModel
# data_list = load_data("2wikimultihopqa", "test", "qwen2.5-1.5b-instruct")

model, tokenizer, _generation_config = get_model("llama3.2-1b-instruct")

adapter_paths = []
num_adapters = 300

for i in range(num_adapters):
    print(f"Loading adapter {i}")
    for pid in range(3):
        adapter_path = f"offline/llama3.2-1b-instruct/rank=2_alpha=32/2wikimultihopqa/lr=0.0003_epoch=1_cot/aug_model=llama3.2-1b-instruct/bridge_comparison/data_{i}/passage_{pid}"
        if i == 0:
            model = PeftModel.from_pretrained(
                                model, 
                                adapter_path,
                                adapter_name = f"passage_{pid}_test_{i}", 
                                is_trainable = False
                            )
        else:
            model.load_adapter(adapter_path, adapter_name = f"passage_{pid}_test_{i}", is_trainable = False)

model.add_weighted_adapter(
    adapters = [f"passage_{pid}_test_{i}" for i in range(num_adapters) for pid in range(3)], 
    weights = [1] * num_adapters * 3,
    adapter_name = "merge", 
    combination_type = "cat",
)

model.set_adapter("merge")

# delete all adapters
for i in range(num_adapters):
    for pid in range(3):
        model.delete_adapter(f"passage_{pid}_test_{i}")

model.save_pretrained("merged_lora_dir/llama3.2-1b-instruct_rank2_alpha32_300")

# adapter_path = "offline/llama3.2-1b-instruct/rank=2_alpha=32/2wikimultihopqa/lr=0.0003_epoch=1_cot/aug_model=llama3.2-1b-instruct/bridge_comparison/data_0/passage_0"

