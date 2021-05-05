from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

from graphics import *
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore', np.RankWarning)

if __name__ == '__main__':
    graphic_builder = GraphicBuilder()
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Task 03')
    plt.grid(True, linestyle='-.')
    plt.title("Сплайн")

    fig.set_size_inches(graphic_builder.WINDOW_WIDTH, graphic_builder.WINDOW_HEIGHT)

    dots_scatter = ax.scatter(graphic_builder.LEARN_DOTS_X, graphic_builder.LEARN_DOTS_Y,
                              s=graphic_builder.DOT_LEARNING_SIZE, color=graphic_builder.DOT_LEARNING_COLOR)

    dots_scatter_test = ax.scatter(graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y,
                                   s=graphic_builder.DOT_TESTING_SIZE, color=graphic_builder.DOT_TESTING_COLOR)

    spline_plot, = ax.plot(graphic_builder.graphic_dots_x, graphic_builder.build_spline_dots(),
                           lw=graphic_builder.LINE_WIDTH,
                           color=graphic_builder.LINE_COLOR)

    fig.tight_layout()

    ax.set_xlim(graphic_builder.GRAPH_MIN_X, graphic_builder.GRAPH_MAX_X)
    ax.set_ylim(graphic_builder.GRAPH_MIN_Y, graphic_builder.GRAPH_MAX_Y)

    ax.legend(
        [Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_LEARNING_COLOR),
         Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_TESTING_COLOR)],
        ("Точки Навчання", "Точки Тестування"),
        numpoints=1)

    dots_axes = plt.axes([0.20, 0.06, 0.65, 0.05], facecolor='blue')
    pow_axes = plt.axes([0.20, 0.02, 0.65, 0.03], facecolor='blue')
    dots_slider = Slider(dots_axes, 'Кількість точок', graphic_builder.MIN_DOTS_AMOUNT, graphic_builder.MAX_DOTS_AMOUNT,
                         valinit=graphic_builder.DOTS_AMOUNT, valstep=graphic_builder.SLIDER_STEP, color='red')
    pow_slider = Slider(pow_axes, 'Степінь сплайну', 1, 3,
                        valinit=1, valstep=1, color='red')

    spline_loss_test = ValueBuilder.least_squares_model(graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y,
                                                        graphic_builder.model)

    dots_slider.label.set_size(12)
    dots_slider.valtext.set_size(14)

    pow_slider.label.set_size(12)
    pow_slider.valtext.set_size(14)

    loss_test_text = fig.text(0.05, 0.95, s=f"Емпіричний ризик(T): {spline_loss_test}", fontsize=14)


    def update_points(val):
        amount_dots = int(dots_slider.val)

        graphic_builder.LEARN_DOTS_X = np.array(ValueBuilder.build_learn_x(amount_dots), float)
        graphic_builder.LEARN_DOTS_Y = np.array(ValueBuilder.build_y(graphic_builder.LEARN_DOTS_X), float)
        xx = np.stack((graphic_builder.LEARN_DOTS_X, graphic_builder.LEARN_DOTS_Y))
        dots_scatter.set_offsets(xx.T)

        graphic_builder.TEST_DOTS_X = np.array(ValueBuilder.build_test_x(amount_dots), float)
        graphic_builder.TEST_DOTS_Y = np.array(ValueBuilder.build_y(graphic_builder.TEST_DOTS_X), float)
        yy = np.stack((graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y))
        dots_scatter_test.set_offsets(yy.T)

        graphic_builder.spline = ValueBuilder.build_spline(pow_slider.val, graphic_builder.LEARN_DOTS_X,
                                                           graphic_builder.LEARN_DOTS_Y)

        spline_plot.set_ydata(graphic_builder.build_spline_dots())

        model_loss_test = ValueBuilder.least_squares_model(graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y,
                                                           graphic_builder.spline)

        loss_test_text.set_text(f"Емпіричний ризик(Т): {model_loss_test}")

        fig.canvas.draw_idle()


    def update_degree(val):
        degree = int(pow_slider.val)
        graphic_builder.spline = ValueBuilder.build_spline(degree, graphic_builder.LEARN_DOTS_X,
                                                           graphic_builder.LEARN_DOTS_Y)
        spline_plot.set_ydata(graphic_builder.build_spline_dots())

        model_loss_test = ValueBuilder.least_squares_model(graphic_builder.TEST_DOTS_X, graphic_builder.TEST_DOTS_Y,
                                                           graphic_builder.spline)

        loss_test_text.set_text(f"Емпіричний ризик(Т): {model_loss_test}")



    dots_slider.on_changed(update_points)
    pow_slider.on_changed(update_degree)

    plt.subplots_adjust(bottom=0.16, right=0.97, top=0.89)
    plt.show()
