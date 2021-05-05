import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, RadioButtons, Button

from src.ValueBuilder import ValueBuilder
from src.GraphicBuilder import GraphicBuilder

if __name__ == "__main__":
    graphic_builder = GraphicBuilder()
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Задача Класифікації')
    plt.title("Задача Класифікації")

    fig.set_size_inches(graphic_builder.WINDOW_WIDTH, graphic_builder.WINDOW_HEIGHT)

    vertical_lines = dict()
    horizontal_lines = dict()

    hor_res, ver_res = graphic_builder.box_hor_ver()

    for index, vertical_line in enumerate(graphic_builder.ab_segment):
        vertical_lines[index] = ax.axvline(vertical_line, linestyle="-.", linewidth=graphic_builder.LINE_WIDTH)

    for index, horizontal_line in enumerate(graphic_builder.cd_segment):
        horizontal_lines[index] = ax.axhline(horizontal_line, linestyle="-.", linewidth=graphic_builder.LINE_WIDTH)

    hor_span = ax.axhspan(graphic_builder.cd_segment[hor_res[0]], graphic_builder.cd_segment[hor_res[1]], color="grey",
                          alpha=0.3)
    ver_span = ax.axvspan(graphic_builder.ab_segment[ver_res[0]], graphic_builder.ab_segment[ver_res[1]], color="grey",
                          alpha=0.3)
    best_dots = ax.scatter(graphic_builder.learning_dots_horizontal, graphic_builder.learning_dots_vertical,
                           s=graphic_builder.DOT_LEARNING_SIZE, color=graphic_builder.DOT_LEARNING_COLOR, zorder=5)

    learning_dot = ax.scatter(graphic_builder.random_point[0], graphic_builder.random_point[1],
                              s=graphic_builder.DOT_TESTING_SIZE, color=graphic_builder.DOT_TESTING_COLOR, zorder=5)

    not_best = ax.scatter(graphic_builder.horizontal_not_best, graphic_builder.vertical_not_best,
                          s=graphic_builder.DOT_NOT_BEST_SIZE, color=graphic_builder.DOT_NOT_BEST_COLOR, zorder=5)

    fig.tight_layout()

    ax.legend(
        [Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_LEARNING_COLOR),
         Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_TESTING_COLOR)],
        ("Точки Навчнання", "Точка Тестування"),
        numpoints=1)

    dots_axes = plt.axes([0.20, 0.03, 0.40, 0.02], facecolor='blue')
    dots_slider = Slider(dots_axes, 'Кількість точок', graphic_builder.MIN_LEARNING_DOTS,
                         graphic_builder.MAX_LEARNING_DOTS,
                         valinit=graphic_builder.l, valstep=10, color='red')

    vert_axes = plt.axes([0.20, 0.09, 0.40, 0.02], facecolor='blue')
    vert_slider = Slider(vert_axes, 'n', graphic_builder.MIN_CLASSES,
                         graphic_builder.MAX_CLASSES,
                         valinit=graphic_builder.n, valstep=1, color='red')

    hor_axes = plt.axes([0.20, 0.06, 0.40, 0.02], facecolor='blue')
    hor_slider = Slider(hor_axes, 'm', graphic_builder.MIN_CLASSES,
                        graphic_builder.MAX_CLASSES,
                        valinit=graphic_builder.m, valstep=1, color='red')

    k_axes = plt.axes([0.20, 0.12, 0.40, 0.02], facecolor='blue')
    k_slider = Slider(k_axes, 'k', graphic_builder.MIN_K, graphic_builder.MAX_K, valinit=graphic_builder.K_NEAREST,
                      valstep=1,
                      color='red')

    percentage_axes = plt.axes([0.20, 0.15, 0.40, 0.02], facecolor='blue')
    percentage_slider = Slider(percentage_axes, 'Еталони', 0.01,
                               1.00, valinit=graphic_builder.PERCENTAGE, valstep=0.01, color='red')

    radio_weight_axes = plt.axes([0.65, 0.03, 0.15, 0.12])
    radio_weight = RadioButtons(radio_weight_axes, ('Без ваги', 'Лінійна', 'Експоненсійна'))

    rax_distance = plt.axes([0.81, 0.03, 0.15, 0.12])
    radio_distance = RadioButtons(rax_distance, ('Евклідова', 'Манхетенна'))

    current_miss = ValueBuilder.classify_miss_by_k(graphic_builder.classified_dots,
                                                   graphic_builder.main_classified,
                                                   graphic_builder.DISTANCE_ALGORITHM,
                                                   graphic_builder.WEIGHT_ALGORITHM,
                                                   graphic_builder.K_NEAREST)

    loss_text = fig.text(0.05, 0.95, s=f"Current hit: {current_miss}/{graphic_builder.l}", fontsize=14)

    best_dot_text = fig.text(0.80, 0.95, s=f"Еталони : {graphic_builder.PERCENTAGE_DOT_AMOUNT}", fontsize=14)

    weight_dict = {'Без ваги': ValueBuilder.no_weigh, 'Лінійна': ValueBuilder.linear_weight,
                   'Експоненсійна': ValueBuilder.exponential_weight}

    distance_dict = {'Евклідова': ValueBuilder.euclidian_distance, 'Манхетенна': ValueBuilder.manhattan_distance}

    button_axes = plt.axes([0.05, 0.14, 0.05, 0.05])
    button = Button(button_axes, 'Навчити K')

    random_axes = plt.axes([0.05, 0.02, 0.05, 0.05])
    random_dot = Button(random_axes, 'Нова Точка')

    calculate_axes = plt.axes([0.05, 0.08, 0.05, 0.05])
    calculate_button = Button(calculate_axes, 'Порахувати\nЯкість')

    button_best_axes = plt.axes([0.11, 0.08, 0.04, 0.05])
    button_best = Button(button_best_axes, 'Навчити \nЕталон')

    button_weigh_axes = plt.axes([0.11, 0.14, 0.04, 0.05])
    button_weigh = Button(button_weigh_axes, 'Навчити \nВагу')


    def calculate_hit(event):
        current_miss = ValueBuilder.classify_miss_by_k(graphic_builder.classified_dots,
                                                       graphic_builder.main_classified,
                                                       graphic_builder.DISTANCE_ALGORITHM,
                                                       graphic_builder.WEIGHT_ALGORITHM,
                                                       graphic_builder.K_NEAREST)
        loss_text.set_text(f"Current hit: {current_miss}/{graphic_builder.l}")


    def update_spans_classify():
        graphic_builder.classify_dot()
        hor_res, ver_res = graphic_builder.box_hor_ver()
        global hor_span, ver_span
        try:
            hor_span.remove()
            ver_span.remove()
        except ValueError:
            pass
        hor_span = ax.axhspan(graphic_builder.cd_segment[hor_res[0]], graphic_builder.cd_segment[hor_res[1]],
                              color="grey",
                              alpha=0.3)
        ver_span = ax.axvspan(graphic_builder.ab_segment[ver_res[0]], graphic_builder.ab_segment[ver_res[1]],
                              color="grey",
                              alpha=0.3)


    def update_percentage(val):
        fig.canvas.draw_idle()
        value = float(val)
        if value == graphic_builder.PERCENTAGE:
            return
        graphic_builder.PERCENTAGE = value
        graphic_builder.get_best_dots()
        best_dot_text.set_text(f"Еталони : {graphic_builder.PERCENTAGE_DOT_AMOUNT}")
        best_dots.set_offsets(graphic_builder.learning_dots)
        graphic_builder.build_difference()
        if len(graphic_builder.not_best_dots) > 0:
            not_best.set_offsets(graphic_builder.not_best_dots)
        update_spans_classify()


    def update_distance(label):
        fig.canvas.draw_idle()
        graphic_builder.DISTANCE_ALGORITHM = distance_dict[label]
        update_spans_classify()


    def update_weight(label):
        fig.canvas.draw_idle()
        graphic_builder.WEIGHT_ALGORITHM = weight_dict[label]
        update_spans_classify()


    def update_k(val):
        fig.canvas.draw_idle()
        graphic_builder.K_NEAREST = int(val)
        update_spans_classify()


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
        update_spans_classify()


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
        update_spans_classify()


    def update_dots(val):
        fig.canvas.draw_idle()
        graphic_builder.l = int(val)
        graphic_builder.generate_dots_and_clasify()
        graphic_builder.get_best_dots()
        best_dot_text.set_text(f"Еталони : {graphic_builder.PERCENTAGE_DOT_AMOUNT}")
        update_spans_classify()
        best_dots.set_offsets(graphic_builder.learning_dots)
        graphic_builder.build_difference()
        if len(graphic_builder.not_best_dots) > 0:
            not_best.set_offsets(graphic_builder.not_best_dots)


    def compute_k(event):
        fig.canvas.draw_idle()
        best_k = ValueBuilder.select_right_k(graphic_builder.classified_dots,
                                             graphic_builder.main_classified,
                                             graphic_builder.DISTANCE_ALGORITHM,
                                             graphic_builder.WEIGHT_ALGORITHM)
        k_slider.set_val(best_k)


    def compute_best_weigh(event):
        fig.canvas.draw_idle()
        weight_index = ValueBuilder.select_best_weigh(graphic_builder.classified_dots,
                                                      graphic_builder.main_classified,
                                                      graphic_builder.DISTANCE_ALGORITHM,
                                                      graphic_builder.WEIGHTS,
                                                      graphic_builder.K_NEAREST)
        radio_weight.set_active(weight_index)
        graphic_builder.WEIGHT_ALGORITHM = graphic_builder.WEIGHTS[weight_index]
        update_spans_classify()


    def learn_best_percentage(event):
        fig.canvas.draw_idle()
        best_percentage = ValueBuilder.find_best_percentage_k(graphic_builder.main_classified,
                                                              graphic_builder.DISTANCE_ALGORITHM,
                                                              graphic_builder.WEIGHT_ALGORITHM,
                                                              graphic_builder.K_NEAREST)
        percentage_slider.set_val(best_percentage)


    def generate_new(event):
        fig.canvas.draw_idle()
        graphic_builder.generate_and_clasify_dot()
        hor_res, ver_res = graphic_builder.box_hor_ver()
        global hor_span, ver_span
        try:
            hor_span.remove()
            ver_span.remove()
        except ValueError:
            pass
        hor_span = ax.axhspan(graphic_builder.cd_segment[hor_res[0]], graphic_builder.cd_segment[hor_res[1]],
                              color="grey",
                              alpha=0.3)
        ver_span = ax.axvspan(graphic_builder.ab_segment[ver_res[0]], graphic_builder.ab_segment[ver_res[1]],
                              color="grey",
                              alpha=0.3)
        learning_dot.set_offsets([graphic_builder.random_point[0], graphic_builder.random_point[1]])


    radio_weight.on_clicked(update_weight)
    radio_distance.on_clicked(update_distance)
    k_slider.on_changed(update_k)
    vert_slider.on_changed(update_n)
    hor_slider.on_changed(update_m)
    dots_slider.on_changed(update_dots)
    button.on_clicked(compute_k)
    calculate_button.on_clicked(calculate_hit)
    random_dot.on_clicked(generate_new)
    button_weigh.on_clicked(compute_best_weigh)
    percentage_slider.on_changed(update_percentage)
    button_best.on_clicked(learn_best_percentage)
    plt.subplots_adjust(bottom=0.235, right=0.98)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
