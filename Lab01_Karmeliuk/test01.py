from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, TextBox

from graphics import *
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np

if __name__ == '__main__':
    plt.show()
    graphic_builder = GraphicBuilder()
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Task 01')
    plt.title("Інтерполяційний поліном")

    fig.set_size_inches(graphic_builder.WINDOW_WIDTH, graphic_builder.WINDOW_HEIGHT)

    dots_scatter = ax.scatter(graphic_builder.LEARN_DOTS_X, graphic_builder.LEARN_DOTS_Y,
                              s=graphic_builder.DOT_LEARNING_SIZE, color=graphic_builder.DOT_LEARNING_COLOR)

    dots_scatter_test = ax.scatter(graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y,
                                   s=graphic_builder.DOT_TESTING_SIZE, color=graphic_builder.DOT_TESTING_COLOR)

    graphic_dots_y = graphic_builder.build_graphic_dots(graphic_builder.LEARN_DOTS_X, graphic_builder.LEARN_DOTS_Y)

    lagrange_plot, = ax.plot(graphic_builder.graphic_dots_x, graphic_dots_y,
                             lw=graphic_builder.LINE_WIDTH,
                             color=graphic_builder.LINE_COLOR)

    sin_plot, = ax.plot(graphic_builder.graphic_dots_x, graphic_builder.sin_dots,
                             lw=graphic_builder.LINE_WIDTH,
                             color="red")

    plt.grid(True, linestyle='-.')
    fig.tight_layout()

    ax.set_xlim(graphic_builder.GRAPH_MIN_X, graphic_builder.GRAPH_MAX_X)
    ax.set_ylim(graphic_builder.GRAPH_MIN_Y, graphic_builder.GRAPH_MAX_Y)

    ax.legend([Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_LEARNING_COLOR)
                  , Line2D(range(1), range(1), color="white", marker='o',
                           markerfacecolor=graphic_builder.DOT_TESTING_COLOR)], ("Точки Навчнання", "Точки Тестування"),
              numpoints=1)

    dots_axes = plt.axes([0.49, 0.04, 0.04, 0.05], facecolor='blue')
    text_box = TextBox(dots_axes, 'Кількість точок', initial=str(graphic_builder.DOTS_AMOUNT),label_pad=0.2)

    model_loss = ValueBuilder.least_squares_lagrange(graphic_builder.LEARN_DOTS_X, graphic_builder.LEARN_DOTS_Y,
                                                     graphic_builder.TEST_DOTS_X,
                                                     graphic_builder.TEST_DOTS_Y)

    def submit(text):
        amount_dots = int(text)
        if amount_dots == graphic_builder.DOTS_AMOUNT:
            return
        graphic_builder.DOTS_AMOUNT = amount_dots

        new_xs = np.array(ValueBuilder.build_learn_x(amount_dots), float)
        new_ys = np.array(ValueBuilder.build_y(new_xs), float)
        xx = np.stack((new_xs, new_ys))
        dots_scatter.set_offsets(xx.T)

        sin_dots = graphic_builder.build_graphic_dots_sin()
        sin_plot.set_ydata(sin_dots)

        graphic_builder.graphic_dots_y = graphic_builder.build_graphic_dots(new_xs, new_ys)
        lagrange_plot.set_ydata(graphic_builder.graphic_dots_y)

        new_test_xs = np.array(ValueBuilder.build_test_x(amount_dots), float)
        new_test_ys = np.array(ValueBuilder.build_y(new_test_xs), float)
        yy = np.stack((new_test_xs, new_test_ys))
        dots_scatter_test.set_offsets(yy.T)

        test_loss = ValueBuilder.least_squares_lagrange(new_xs, new_ys, new_test_xs, new_test_ys)
        loss_text.set_text(f"Model test loss: {test_loss}")
        plt.draw()


    text_box.on_submit(submit)

    loss_text = fig.text(0.05, 0.95, s=f"Model test loss: {model_loss}", fontsize=14)

    plt.subplots_adjust(bottom=0.155, right=0.98)
    plt.show()
