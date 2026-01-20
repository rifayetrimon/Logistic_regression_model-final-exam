import numpy as np
import pandas as pd
import gradio as gr
import pickle


with open('lg_reg_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)


def predict_price_range(
    battery_power, blue, clock_speed, dual_sim, four_g,
    int_memory, m_dep, mobile_wt, n_cores, pc,
    px_height, px_width, ram, sc_h, sc_w, talk_time, touch_screen, wifi
    ):
    
    input_df = pd.DataFrame([{
        'battery_power': battery_power,
        'blue': blue,
        'clock_speed': clock_speed,
        'dual_sim': dual_sim,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'touch_screen': touch_screen,
        'wifi': wifi
    }])
    
    prediction = model.predict(input_df)[0]    
    
    if prediction == 0:
        return 'Low'
    elif prediction == 1:
        return 'Medium'
    elif prediction == 2:
        return 'High'
    else:
        return 'Very High'
    
    

inputs = [
    gr.Slider(minimum=501, maximum=1998, step=1, label='battery_power'),
    gr.Checkbox(label='blue'),
    gr.Slider(minimum=0.5, maximum=3, step=0.1, label='clock_speed'),
    gr.Checkbox(label='dual_sim'),
    gr.Checkbox(label='four_g'),
    gr.Slider(minimum=2, maximum=64, step=1, label='int_memory'),
    gr.Slider(minimum=0.1, maximum=1, step=0.01, label='m_dep'),
    gr.Slider(minimum=80, maximum=200, step=1, label='mobile_wt'),
    gr.Slider(minimum=1, maximum=8, step=1, label='n_cores'),
    gr.Slider(minimum=0, maximum=20, step=1, label='pc'),
    gr.Slider(minimum=700, maximum=1960, step=1, label='px_height'),
    gr.Slider(minimum=500, maximum=1998, step=1, label='px_width'),
    gr.Slider(minimum=256, maximum=3998, step=1, label='ram'),
    gr.Slider(minimum=5, maximum=19, step=1, label='sc_h'),
    gr.Slider(minimum=1, maximum=18, step=1, label='sc_w'),
    gr.Slider(minimum=2, maximum=20, step=1, label='talk_time'),
    gr.Checkbox(label='touch_screen'),
    gr.Checkbox(label='wifi')
]


app = gr.Interface(
    fn=predict_price_range,
    inputs=inputs,
    outputs='text',
    title='Mobile Price Range Predictor',
    description='Enter the features of the mobile and predict the price range'
)

app.launch()