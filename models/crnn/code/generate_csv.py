import json
import csv
import os

def generate_csv(pred_file, test_dir, output_file):
    # 读取预测结果
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    # 获取所有测试图片并排序
    all_images = sorted(
        [f for f in os.listdir(test_dir) if f.endswith('.png')],
        key=lambda x: int(x.split('.')[0])
    )
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 写入CSV文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['file_name', 'file_code'])
        
        # 写入每一行数据
        processed_count = 0
        empty_count = 0
        
        for img_name in all_images:
            if img_name in predictions:
                # 将标签列表转换为字符串
                code = ''.join([str(l) for l in predictions[img_name]['label']]) if predictions[img_name]['label'] else ''
                processed_count += 1
            else:
                code = ''
                empty_count += 1
            writer.writerow([img_name, code])
    
    print(f'CSV文件已生成：{output_file}')
    print(f'总图片数：{len(all_images)}')
    print(f'有预测结果：{processed_count} 张')
    print(f'无预测结果：{empty_count} 张')

if __name__ == '__main__':
    pred_file = '../prediction_result/predictions.json'
    test_dir = '../tcdata/mchar_test_a'
    output_file = '../prediction_result/result.csv'
    
    generate_csv(pred_file, test_dir, output_file) 