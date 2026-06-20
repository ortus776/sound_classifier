import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Настройка стилей для научной публикации (шрифт без засечек, чистые цвета)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(14, 3), dpi=300)

# Определение блоков архитектуры: (Название, Описание слоев, Размерность на выходе)
blocks = [
    ("Вход", "Mel-Sg\nStereo", "2×64×345"),
    ("Блок 1", "2 × [Conv 3x3, 16]\nBatchNorm\nMaxPool 2x2", "16×32×172"),
    ("Блок 2", "2 × [Conv 3x3, 32]\nBatchNorm\nMaxPool 2x2", "32×16×86"),
    ("Блок 3", "2 × [Conv 3x3, 64]\nBatchNorm\nMaxPool 2x2", "64×8×43"),
    ("Блок 4", "2 × [Conv 3x3, 128]\nBatchNorm\nMaxPool 2x2", "128×4×21"),
    ("Пулинг", "Adaptive\nAvgPool\nFlatten", "128"),
    ("Голова", "Linear(128)\nDropout(0.3)\nLinear(10)", "10")
]

box_width = 1.4
box_height = 1.6
spacing = 0.5
start_x = 0.5
y_pos = 0.5

# Отрисовка блоков цепочки
for i, (title, layer_desc, shape) in enumerate(blocks):
    x_pos = start_x + i * (box_width + spacing)
    
    # Подбор цвета: вход/выход серые, скрытые блоки — градиент сине-зеленого
    if i in [0, 6]:
        facecolor = '#f1f3f5'
    else:
        facecolor = f'#e3faf2' if i % 2 == 0 else '#e7f5ff'
        
    # Рисуем прямоугольник блока
    rect = patches.FancyBboxPatch(
        (x_pos, y_pos), box_width, box_height, 
        boxstyle="round,pad=0.03", linewidth=1.5, 
        edgecolor='#495057', facecolor=facecolor
    )
    ax.add_patch(rect)
    
    # Текст внутри блока
    plt.text(x_pos + box_width/2, y_pos + 1.3, title, weight='bold', ha='center', va='center', color='#212529')
    plt.text(x_pos + box_width/2, y_pos + 0.7, layer_desc, ha='center', va='center', color='#495057', fontsize=8)
    
    # Текст размерности тензора под блоком
    plt.text(x_pos + box_width/2, y_pos - 0.2, shape, ha='center', va='center', weight='semibold', color='#1c7ed6', fontsize=9)
    
    # Стрелка к следующему блоку
    if i < len(blocks) - 1:
        arrow_start_x = x_pos + box_width
        arrow_end_x = arrow_start_x + spacing
        ax.annotate('', xy=(arrow_end_x, y_pos + box_height/2), xytext=(arrow_start_x, y_pos + box_height/2),
                    arrowprops=dict(arrowstyle="->", color='#868e96', lw=2, mutation_scale=15))

# Очистка осей координат
ax.set_xlim(0, start_x + len(blocks) * (box_width + spacing) - spacing + 0.5)
ax.set_ylim(-0.2, y_pos + box_height + 0.3)
ax.axis('off')

plt.tight_layout()
plt.savefig("compact_scientific_architecture.png", bbox_inches='tight', transparent=False)
print("Компактный график сохранен как compact_scientific_architecture.png")