import json

# 读取预测结果
with open('../prediction_result/predictions.json', 'r') as f:
    predictions = json.load(f)

# 显示第一个预测结果的详细信息
first_img = list(predictions.keys())[0]
print(f"第一张图片 {first_img} 的预测结果：")
print(json.dumps(predictions[first_img], indent=2, ensure_ascii=False))

# 显示前5张图片的边界框数量
print("\n前5张图片的边界框数量：")
for i, (img_name, pred) in enumerate(list(predictions.items())[:5]):
    print(f"{img_name}: {len(pred['left'])} 个边界框，标签：{pred['label']}") 