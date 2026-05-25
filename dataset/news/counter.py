def count_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    return line_count

filepath = "news_full.txt"
print(f"There are {count_lines(filepath)} lines in the file:{filepath}")
