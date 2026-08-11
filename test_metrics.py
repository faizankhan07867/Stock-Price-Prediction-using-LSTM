import numpy as np

from utils.metrics import *

actual = np.array([100,102,101,105,110])

predicted = np.array([101,101,102,106,109])

evaluate(actual,predicted)

save_prediction_plot(

    actual,

    predicted

)

save_loss_plot(

    [0.6,0.4,0.3,0.2],

    [0.7,0.5,0.35,0.25]

)