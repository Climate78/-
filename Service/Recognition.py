import os

from dotenv import load_dotenv

from . import read_one_img_to_matrix, neuron_net


def recognition(img_path):
    img_matrix = read_one_img_to_matrix(img_path)

    load_dotenv()
    scales_index = int(os.getenv('scales_index'))
    layer_matrices = img_matrix.iloc[0]

    recognizer = neuron_net(layer_matrices, scales_index)[0]
    get_answer = recognizer.idxmax()
    services = {
        0: "iMessage",
        1: "ICQ",
        2: "Viber",
        3: "WeChat",
        4: "WhatsApp",
        5: "Google Talk",
        6: "Line",
        7: "Snapchat",
        8: "Telegram",
        9: "Facebook"
    }

    return services.get(get_answer, "Invalid value. Please enter a number from 0 to 9.")
