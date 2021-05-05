import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
from matplotlib.patches import Rectangle
from src.ValueBuilder import ValueBuilder
from src.GraphicBuilder import GraphicBuilder

if __name__ == "__main__":
    graphic_builder = GraphicBuilder(tree=True)
    fig, ax = plt.subplots()

    manager = plt.get_current_fig_manager()
    manager.set_window_title('Побудова')
    plt.title("Задача логічної класифікації")

    fig.set_size_inches(graphic_builder.WINDOW_WIDTH, graphic_builder.WINDOW_HEIGHT)

    vertical_lines = dict()
    horizontal_lines = dict()

    for index, vertical_line in enumerate(graphic_builder.ab_segment):
        vertical_lines[index] = ax.axvline(vertical_line, linestyle="-.", linewidth=graphic_builder.LINE_WIDTH)

    for index, horizontal_line in enumerate(graphic_builder.cd_segment):
        horizontal_lines[index] = ax.axhline(horizontal_line, linestyle="-.", linewidth=graphic_builder.LINE_WIDTH)

    learning_dots = ax.scatter(graphic_builder.learning_dots_horizontal, graphic_builder.learning_dots_vertical,
                               s=graphic_builder.DOT_LEARNING_SIZE, color=graphic_builder.DOT_LEARNING_COLOR, zorder=5)

    testing_dots = ax.scatter(list(map(lambda x: x[0], graphic_builder.testing_dots)),
                              list(map(lambda x: x[1], graphic_builder.testing_dots)),
                              s=graphic_builder.DOT_TESTING_SIZE, color=graphic_builder.DOT_TESTING_COLOR, zorder=5)

    random_dot_scatter = ax.scatter(graphic_builder.random_dot[0], graphic_builder.random_dot[1],
                                    s=graphic_builder.RANDOM_DOT_SIZE, color=graphic_builder.RANDOM_DOT_COLOR, zorder=5)

    rect = Rectangle((0, 0), 0, 0, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    fig.tight_layout()

    ax.legend(
        [Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_LEARNING_COLOR),
         Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_TESTING_COLOR)],
        ("Точки Навчнання", "Точка Тестування"),
        numpoints=1)

    dots_axes = plt.axes([0.36, 0.93, 0.60, 0.02], facecolor='blue')
    dots_slider = Slider(dots_axes, 'Кількість точок', graphic_builder.MIN_LEARNING_DOTS,
                         graphic_builder.MAX_LEARNING_DOTS,
                         valinit=graphic_builder.l, valstep=10, color='red')

    hor_axes = plt.axes([0.36, 0.90, 0.60, 0.02], facecolor='blue')
    hor_slider = Slider(hor_axes, 'm', graphic_builder.MIN_CLASSES,
                        graphic_builder.MAX_CLASSES,
                        valinit=graphic_builder.m, valstep=1, color='red')

    vert_axes = plt.axes([0.36, 0.87, 0.60, 0.02], facecolor='blue')
    vert_slider = Slider(vert_axes, 'n', graphic_builder.MIN_CLASSES,
                         graphic_builder.MAX_CLASSES,
                         valinit=graphic_builder.n, valstep=1, color='red')

    zones_axes = plt.axes([0.05, 0.90, 0.08, 0.05])
    zones_button = Button(zones_axes, 'Побудувати зони')

    tree_axes = plt.axes([0.05, 0.84, 0.08, 0.05])
    tree_button = Button(tree_axes, 'Побудувати Дерево')

    list_axes = plt.axes([0.15, 0.84, 0.08, 0.05])
    list_button = Button(list_axes, 'Побудувати Список')

    random_axes = plt.axes([0.15, 0.90, 0.08, 0.05])
    random_dot = Button(random_axes, 'Нова Точка')

    hit_text = fig.text(0.05, 0.77, s=f"", fontsize=16)


    def update_n(val):
        fig.canvas.draw_idle()
        graphic_builder.n = int(val)
        graphic_builder.generate_ab_and_classify()
        global vertical_lines
        for vertical_line in list(vertical_lines):
            vertical_lines[vertical_line].remove()
            vertical_lines.pop(vertical_line, None)
        for index, vertical_line in enumerate(graphic_builder.ab_segment):
            vertical_lines[index] = ax.axvline(vertical_line, linestyle="-.", linewidth=graphic_builder.LINE_WIDTH)


    def update_m(val):
        fig.canvas.draw_idle()
        graphic_builder.m = int(val)
        graphic_builder.generate_cd_and_classify()
        global horizontal_lines
        for hor_line in list(horizontal_lines):
            horizontal_lines[hor_line].remove()
            horizontal_lines.pop(hor_line, None)
        for index, horizontal_line in enumerate(graphic_builder.cd_segment):
            horizontal_lines[index] = ax.axhline(horizontal_line, linestyle="-.", linewidth=graphic_builder.LINE_WIDTH)


    def update_dots(val):
        fig.canvas.draw_idle()
        graphic_builder.l = int(val)
        graphic_builder.generate_dots_and_clasify()
        learning_dots.set_offsets(ValueBuilder.convert_to_list(graphic_builder.classified_dots))
        testing_dots.set_offsets(ValueBuilder.convert_to_list(graphic_builder.testing_dots))


    def calculate_hit():
        hit_text.set_text(f"{graphic_builder.calculate_hit()}/{len(graphic_builder.testing_dots)}")


    def generate_new(event):
        fig.canvas.draw_idle()
        graphic_builder.generate_random_dot()
        random_dot_scatter.set_offsets([graphic_builder.random_dot[0], graphic_builder.random_dot[1]])
        graphic_builder.selected_class = graphic_builder.classification_method(graphic_builder.random_dot)
        if graphic_builder.selected_class is not None:
            (x, y), width, height = graphic_builder.get_ab_cd_points(graphic_builder.selected_class)
        else:
            x, y = 0, 0
            width = 0
            height = 0
        global rect
        rect.set_xy((x, y))
        rect.set_height(height)
        rect.set_width(width)


    def generate_tree(event):
        graphic_builder.build_tree()
        graphic_builder.classification_method = graphic_builder.apply_tree
        calculate_hit()


    def generate_list(event):
        graphic_builder.build_list()
        graphic_builder.classification_method = graphic_builder.apply_list
        calculate_hit()


    def build_zones(evnet):
        graphic_builder.build_zones()


    vert_slider.on_changed(update_n)
    hor_slider.on_changed(update_m)
    dots_slider.on_changed(update_dots)

    zones_button.on_clicked(build_zones)
    random_dot.on_clicked(generate_new)
    tree_button.on_clicked(generate_tree)
    list_button.on_clicked(generate_list)

    plt.subplots_adjust(top=0.73, right=0.98)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
