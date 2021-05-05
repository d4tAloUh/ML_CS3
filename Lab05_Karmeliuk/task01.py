import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
from matplotlib.patches import Rectangle
from src.ValueBuilder import ValueBuilder
from src.GraphicBuilder import GraphicBuilder

if __name__ == "__main__":
    graphic_builder = GraphicBuilder()
    fig, ax = plt.subplots()

    manager = plt.get_current_fig_manager()
    manager.set_window_title('Порівяння інформативності')
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

    # testing_dots = ax.scatter(list(map(lambda x: x[0], graphic_builder.testing_dots)),
    #                           list(map(lambda x: x[1], graphic_builder.testing_dots)),
    #                           s=graphic_builder.DOT_TESTING_SIZE, color=graphic_builder.DOT_TESTING_COLOR, zorder=5)

    (x, y), width, height = graphic_builder.get_ab_cd_points(graphic_builder.selected_class)
    rect = Rectangle((x, y), width, height, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    fig.tight_layout()

    ax.legend(
        [Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_LEARNING_COLOR),
         Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_TESTING_COLOR)],
        ("Точки Навчнання", "Точка Тестування"),
        numpoints=1)

    dots_axes = plt.axes([0.36, 0.95, 0.60, 0.02], facecolor='blue')
    dots_slider = Slider(dots_axes, 'Кількість точок', graphic_builder.MIN_LEARNING_DOTS,
                         graphic_builder.MAX_LEARNING_DOTS,
                         valinit=graphic_builder.l, valstep=10, color='red')

    hor_axes = plt.axes([0.36, 0.92, 0.60, 0.02], facecolor='blue')
    hor_slider = Slider(hor_axes, 'm', graphic_builder.MIN_CLASSES,
                        graphic_builder.MAX_CLASSES,
                        valinit=graphic_builder.m, valstep=1, color='red')

    vert_axes = plt.axes([0.36, 0.89, 0.60, 0.02], facecolor='blue')
    vert_slider = Slider(vert_axes, 'n', graphic_builder.MIN_CLASSES,
                         graphic_builder.MAX_CLASSES,
                         valinit=graphic_builder.n, valstep=1, color='red')

    vert_eps_axes = plt.axes([0.36, 0.86, 0.60, 0.02], facecolor='blue')
    vert_eps_slider = Slider(vert_eps_axes, 'x2 eps', -1.0,
                             1.0,
                             valinit=graphic_builder.eps_vert,
                             valstep=0.1, color='red')

    hor_eps_axes = plt.axes([0.36, 0.83, 0.60, 0.02], facecolor='blue')
    hor_eps_slider = Slider(hor_eps_axes, 'x1 eps', -1.0,
                            1.0,
                            valinit=graphic_builder.eps_hor,
                            valstep=0.1, color='red')

    class_axes = plt.axes([0.36, 0.80, 0.05, 0.02], facecolor='blue')
    class_box = TextBox(class_axes, 'Класс', initial=graphic_builder.selected_class)

    heuristic, statistic, entropy, compare = graphic_builder.calculate_informativeness()
    heuristic_info = fig.text(0.02, 0.95, s=f"Heuristic: {heuristic}", fontsize=12)
    statistical_info = fig.text(0.02, 0.91, s=f"Statistical: {statistic}", fontsize=12)
    entropy_info = fig.text(0.02, 0.87, s=f"Entropy: {entropy}", fontsize=12)
    # compare_info = fig.text(0.02, 0.83, s=f"Comparing: {compare}", fontsize=12)


    def update_informativeness():
        heuristic, statistic, entropy, compare = graphic_builder.calculate_informativeness()
        heuristic_info.set_text(f"Heuristic: {heuristic}")
        statistical_info.set_text(f"Statistical: {statistic}")
        entropy_info.set_text(f"Entropy: {entropy}")
        # compare_info.set_text(f"Comparing: {compare}")


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
        update_rect()


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
        update_rect()


    def update_dots(val):
        fig.canvas.draw_idle()
        graphic_builder.l = int(val)
        graphic_builder.generate_dots_and_clasify()
        learning_dots.set_offsets(ValueBuilder.convert_to_list(graphic_builder.classified_dots))
        # testing_dots.set_offsets(ValueBuilder.convert_to_list(graphic_builder.testing_dots))
        update_informativeness()


    def update_rect():
        (x, y), width, height = graphic_builder.get_ab_cd_points(graphic_builder.selected_class)
        global rect
        rect.set_xy((x, y))
        rect.set_height(height)
        rect.set_width(width)
        update_informativeness()


    def update_hor_eps(val):
        fig.canvas.draw_idle()
        graphic_builder.eps_hor = float(val)
        update_rect()


    def update_vert_eps(val):
        fig.canvas.draw_idle()
        graphic_builder.eps_vert = float(val)
        update_rect()


    def update_class(text):
        fig.canvas.draw_idle()
        class_num = int(text)
        graphic_builder.selected_class = class_num
        update_rect()


    vert_slider.on_changed(update_n)
    hor_slider.on_changed(update_m)
    dots_slider.on_changed(update_dots)
    hor_eps_slider.on_changed(update_hor_eps)
    vert_eps_slider.on_changed(update_vert_eps)
    class_box.on_submit(update_class)
    plt.subplots_adjust(top=0.73, right=0.98)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
