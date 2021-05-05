from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, TextBox

from graphics import *
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore', np.RankWarning)
current_degree = 1

if __name__ == '__main__':
    graphic_builder = GraphicBuilder()
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Task 02')
    plt.grid(True, linestyle='-.')
    plt.title("Модель")

    fig.set_size_inches(graphic_builder.WINDOW_WIDTH, graphic_builder.WINDOW_HEIGHT)

    dots_scatter = ax.scatter(graphic_builder.LEARN_DOTS_X, graphic_builder.LEARN_DOTS_Y,
                              s=graphic_builder.DOT_LEARNING_SIZE, color=graphic_builder.DOT_LEARNING_COLOR)

    dots_scatter_test = ax.scatter(graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y,
                                   s=graphic_builder.DOT_TESTING_SIZE, color=graphic_builder.DOT_TESTING_COLOR)

    model_plot, = ax.plot(graphic_builder.graphic_dots_x, graphic_builder.model(graphic_builder.graphic_dots_x),
                          lw=graphic_builder.LINE_WIDTH,
                          color=graphic_builder.LINE_COLOR)

    sin_plot, = ax.plot(graphic_builder.graphic_dots_x, graphic_builder.sin_dots,
                             lw=graphic_builder.LINE_WIDTH,
                             color="red")

    fig.tight_layout()
    print(graphic_builder.model)
    ax.set_xlim(graphic_builder.GRAPH_MIN_X, graphic_builder.GRAPH_MAX_X)
    ax.set_ylim(graphic_builder.GRAPH_MIN_Y, graphic_builder.GRAPH_MAX_Y)

    ax.legend(
        [Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_LEARNING_COLOR),
         Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_TESTING_COLOR)],
        ("Точки Навчання", "Точки Тестування"),
        numpoints=1)

    dots_axes = plt.axes([0.20, 0.04, 0.04, 0.05], facecolor='blue')
    pow_axes = plt.axes([0.60, 0.04, 0.04, 0.05], facecolor='blue')
    dots_box = TextBox(dots_axes, 'Кількість точок', initial=str(graphic_builder.DOTS_AMOUNT), label_pad=0.2)
    pow_box = TextBox(pow_axes, 'Степінь', initial=str(1), label_pad=0.2)

    model_loss_learn = ValueBuilder.least_squares_model(graphic_builder.LEARN_DOTS_X, graphic_builder.LEARN_DOTS_Y,
                                                        graphic_builder.model)
    model_loss_test = ValueBuilder.least_squares_model(graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y,
                                                       graphic_builder.model)

    loss_test_text = fig.text(0.05, 0.95, s=f"Емпіричний ризик(T): {model_loss_test}", fontsize=14)
    loss_learn_text = fig.text(0.05, 0.91, s=f"Емпіричний ризик(L): {model_loss_learn}", fontsize=14)


    def update_points(text):
        amount_dots = int(text)
        if amount_dots == graphic_builder.DOTS_AMOUNT:
            return
        graphic_builder.DOTS_AMOUNT = amount_dots

        graphic_builder.LEARN_DOTS_X = np.array(ValueBuilder.build_learn_x(amount_dots), float)
        graphic_builder.LEARN_DOTS_Y = np.array(ValueBuilder.build_y(graphic_builder.LEARN_DOTS_X), float)
        xx = np.stack((graphic_builder.LEARN_DOTS_X, graphic_builder.LEARN_DOTS_Y))
        dots_scatter.set_offsets(xx.T)

        graphic_builder.TEST_DOTS_X = np.array(ValueBuilder.build_test_x(amount_dots), float)
        graphic_builder.TEST_DOTS_Y = np.array(ValueBuilder.build_y(graphic_builder.TEST_DOTS_X), float)
        yy = np.stack((graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y))
        dots_scatter_test.set_offsets(yy.T)

        graphic_builder.model = np.poly1d(
            ValueBuilder.build_polynom_model(int(pow_box.text), graphic_builder.LEARN_DOTS_X,
                                             graphic_builder.LEARN_DOTS_Y))
        print(
            "--------------------------------------------------------------------------------------------------------")
        print(graphic_builder.model)
        print(
            "--------------------------------------------------------------------------------------------------------")
        model_dots = graphic_builder.build_model_dots()
        model_plot.set_ydata(model_dots)

        model_loss_learn = ValueBuilder.least_squares_model(graphic_builder.LEARN_DOTS_X, graphic_builder.LEARN_DOTS_Y,
                                                            graphic_builder.model)
        model_loss_test = ValueBuilder.least_squares_model(graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y,
                                                           graphic_builder.model)

        loss_test_text.set_text(f"Емпіричний ризик(Т): {model_loss_test}")
        loss_learn_text.set_text(f"Емпіричний ризик(Н): {model_loss_learn}")
        plt.draw()


    def update_degree(text):
        global current_degree
        degree = int(text)
        if degree == current_degree:
            return
        current_degree = degree
        graphic_builder.model = np.poly1d(ValueBuilder.build_polynom_model(degree, graphic_builder.LEARN_DOTS_X,
                                                                           graphic_builder.LEARN_DOTS_Y))
        print(
            "--------------------------------------------------------------------------------------------------------")
        print(graphic_builder.model)
        print(
            "--------------------------------------------------------------------------------------------------------")
        model_dots = graphic_builder.build_model_dots()
        model_plot.set_ydata(model_dots)

        model_loss_learn = ValueBuilder.least_squares_model(graphic_builder.LEARN_DOTS_X, graphic_builder.LEARN_DOTS_Y,
                                                            graphic_builder.model)
        model_loss_test = ValueBuilder.least_squares_model(graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y,
                                                           graphic_builder.model)

        loss_test_text.set_text(f"Емпіричний ризик(Т) : {model_loss_test}")
        loss_learn_text.set_text(f"Емпіричний ризик(Н): {model_loss_learn}")
        plt.draw()


    dots_box.on_submit(update_points)
    pow_box.on_submit(update_degree)

    plt.subplots_adjust(bottom=0.16, right=0.97, top=0.89)
    plt.show()
