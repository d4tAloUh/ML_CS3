from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, RadioButtons, Button

from graphics import *
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore', np.RankWarning)

if __name__ == '__main__':
    graphic_builder = GraphicBuilder(sliding=True)
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Sliding')
    plt.grid(True, linestyle='-.')
    plt.title("Надарай-Ватсон")

    fig.set_size_inches(graphic_builder.WINDOW_WIDTH, graphic_builder.WINDOW_HEIGHT)

    dots_scatter = ax.scatter(graphic_builder.LEARNING_DOTS_X, graphic_builder.LEARNING_DOTS_Y,
                              s=graphic_builder.DOT_LEARNING_SIZE, color=graphic_builder.DOT_LEARNING_COLOR, zorder=5)

    dots_scatter_test = ax.scatter(graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y,
                                   s=graphic_builder.DOT_TESTING_SIZE, color=graphic_builder.DOT_TESTING_COLOR,
                                   zorder=7)

    fig.tight_layout()

    ax.set_xlim(graphic_builder.GRAPH_MIN_X, graphic_builder.GRAPH_MAX_X)
    ax.set_ylim(graphic_builder.GRAPH_MIN_Y - graphic_builder.EXTREMAL_VALUE, graphic_builder.GRAPH_MAX_Y + graphic_builder.EXTREMAL_VALUE)

    ax.legend(
        [Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_LEARNING_COLOR),
         Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_TESTING_COLOR)],
        ("Точки Навчання", "Точки Тестування"),
        numpoints=1)

    learning_dots_axes = plt.axes([0.10, 0.03, 0.60, 0.02], facecolor='blue')
    learning_dots_slider = Slider(learning_dots_axes, 'Точки Навчання', graphic_builder.MIN_LEARNING_DOTS_AMOUNT,
                                  graphic_builder.MAX_LEARNING_DOTS_AMOUNT,
                                  valinit=graphic_builder.LEARNING_DOTS_AMOUNT, valstep=graphic_builder.SLIDER_STEP,
                                  color='red')

    test_dots_axes = plt.axes([0.10, 0.06, 0.60, 0.02], facecolor='blue')
    test_dots_slider = Slider(test_dots_axes, 'Точки Тестування', graphic_builder.MIN_TEST_DOTS_AMOUNT,
                              graphic_builder.MAX_TEST_DOTS_AMOUNT,
                              valinit=graphic_builder.TEST_DOTS_AMOUNT, valstep=graphic_builder.SLIDER_STEP,
                              color='red')

    k_neighbours_axes = plt.axes([0.10, 0.09, 0.60, 0.02], facecolor='blue')
    k_neighbours_slider = Slider(k_neighbours_axes, 'Номер сусіду', graphic_builder.MIN_SLIDING_NEIGHBOUR_NUM,
                                 graphic_builder.MAX_SLIDING_NEIGHBOUR_NUM, valinit=graphic_builder.neighbour_sliding,
                                 valstep=1, color='red')

    sin_plot, = ax.plot(graphic_builder.graphic_dots_x, graphic_builder.graphic_dots_y,
                        lw=graphic_builder.LINE_WIDTH,
                        color="red", zorder=3)

    kernel_dict = {'Прямокутна': ValueBuilder.kernel_rect, 'Трикутна': ValueBuilder.kernel_triangle,
                   'Квадратична': ValueBuilder.kernel_square, "Гаусова": ValueBuilder.kernel_gauss}

    radio_kernel_axes = plt.axes([0.85, 0.03, 0.12, 0.09])
    radio_kernel = RadioButtons(radio_kernel_axes, ('Прямокутна', 'Трикутна', 'Квадратична', "Гаусова"))
    current_loss = ValueBuilder.classify_miss_by_best(graphic_builder.learning_dots, graphic_builder.DISTANCE_METHOD,
                                                      graphic_builder.KERNEL_FUNCTION,
                                                      graphic_builder.neighbour_sliding, sliding=True)
    loss_text = fig.text(0.05, 0.95, s=f"Current loss: {current_loss}", fontsize=14)

    button_learn_k_axes = plt.axes([0.75, 0.07, 0.05, 0.05])
    button_learn_k = Button(button_learn_k_axes, 'Навчити \nK')

    button_learn_kernel_axes = plt.axes([0.75, 0.01, 0.05, 0.05])
    button_learn_kernel = Button(button_learn_kernel_axes, 'Навчити \nЯдро')


    def count_loss():
        current_loss = ValueBuilder.classify_miss_by_best(graphic_builder.learning_dots,
                                                          graphic_builder.DISTANCE_METHOD,
                                                          graphic_builder.KERNEL_FUNCTION,
                                                          graphic_builder.neighbour_sliding, sliding=True)
        loss_text.set_text(f"Current loss: {current_loss}")


    def update_learning_dots(val):
        fig.canvas.draw_idle()
        graphic_builder.LEARNING_DOTS_AMOUNT = int(val)
        graphic_builder.generate_dots()
        dots_scatter.set_offsets(graphic_builder.learning_dots[:,[0,1]])
        update_testing_dots()
        # updating slider
        k_neighbours_slider.valmax = graphic_builder.LEARNING_DOTS_AMOUNT - 2
        k_neighbours_slider.ax.set_xlim(k_neighbours_slider.valmin, k_neighbours_slider.valmax)


    def update_testing_dots():
        graphic_builder.get_values_for_x_dots()
        dots_scatter_test.set_offsets(graphic_builder.TEST_DOTS)
        count_loss()


    def update_test_dots(val):
        fig.canvas.draw_idle()
        graphic_builder.TEST_DOTS_AMOUNT = int(val)
        graphic_builder.generate_random_x()
        update_testing_dots()


    def update_neighbour_number(val):
        graphic_builder.neighbour_sliding = int(val)
        update_testing_dots()


    def update_kernel(label):
        fig.canvas.draw_idle()
        graphic_builder.KERNEL_FUNCTION = kernel_dict[label]
        update_testing_dots()


    def learn_k(event):
        k_neighbours = ValueBuilder.learn_best_neighbours(graphic_builder.learning_dots,
                                                          graphic_builder.DISTANCE_METHOD,
                                                          graphic_builder.KERNEL_FUNCTION)
        print(f"Result = {k_neighbours}")
        k_neighbours_slider.set_val(k_neighbours)


    def learn_kernel(event):
        kernel_index = ValueBuilder.select_best_kernel(graphic_builder.learning_dots,
                                                       graphic_builder.DISTANCE_METHOD,
                                                       graphic_builder.kernels,
                                                       graphic_builder.neighbour_sliding,
                                                       sliding=True)
        print(f"Result index = {kernel_index}")
        radio_kernel.set_active(kernel_index)
        graphic_builder.KERNEL_ALGORITHM = graphic_builder.kernels[kernel_index]


    learning_dots_slider.on_changed(update_learning_dots)
    test_dots_slider.on_changed(update_test_dots)
    k_neighbours_slider.on_changed(update_neighbour_number)
    radio_kernel.on_clicked(update_kernel)
    button_learn_kernel.on_clicked(learn_kernel)
    button_learn_k.on_clicked(learn_k)
    plt.subplots_adjust(bottom=0.16, right=0.97, top=0.89)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
