import inspect
import re


def extract_response(text: str):
    parts = text.split("Response:")
    if len(parts) > 1:
        return inspect.cleandoc("".join(parts[1:]).strip())
    else:
        return text


def extract_res_format(texto: str):
    patron = re.compile(r"\[RES\](.*?)\[/RES\]", re.DOTALL)
    resultados = patron.findall(texto)
    if len(resultados) > 0:
        return resultados[0]
    else:
        return texto


def clean_response(text: str):
    text = text.replace("[RES]", "")
    text = text.replace("[/RES]", "")
    text = text.replace("</s>", "")
    text = text.replace("<s>", "")
    text = text.replace("[INST]", "")
    text = text.replace("[/INST]", "")
    text = text.replace("<<<SYS>>>", "")
    text = text.replace("[/", "")
    return text


def get_answer(text: str):
    text = extract_response(text)
    text = clean_response(text)
    text = extract_res_format(text)
    text = inspect.cleandoc(text.strip())
    return text


if __name__ == "__main__":

    text1 = inspect.cleandoc(
        """
        [INST] <<<SYS>>> You are an expert in any topic, please response the following  <<<SYS>>> Program me a bubble sort algorithm in python [/INST] Response: Sure, I'd be happy to help you with that! Here is a basic implementation of the bubble sort algorithm in Python:
        everybody loves bubble sort! 

        def bubble_sort(arr):
            n = len(arr)
            for i in range(n-1):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        # swap the two elements
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr

        Here's how the algorithm works:

        1. We loop through the array `arr` from the last index to the first index.
        2. Within each loop, we loop through the array from the first index to the last index, minus the current index (i).
    """
    )

    res1 = get_answer(text1)
    print(res1)

    print()

    text2 = inspect.cleandoc(
        """
        [INST] You are an expert assistant in R for SIVIREP, a library for epidemiological data analysis. Provide accurate, clear R code and explanations for technical queries. Keep responses concise, structured, and relevant, with a focus on precision. Get cases of Severe Dengue by cod_eve [/INST] Response: [RES]  To extract cases of Severe Dengue using the cod_eve variable from a data frame named `dengue_data`, you can use the following R code: ```R # Assuming 'dengue_data' is your data frame severe_dengue <- dengue_data %>% filter(cod_eve == "Severe Dengue") ``` This code uses the `dplyr` package to filter the rows in the `dengue_data` data frame where the `cod_eve` variable equals "Severe Dengue". The resulting data frame, `severe_dengue`, will contain only the rows with this condition. </s>
        """
    )

    res2 = get_answer(text2)
    print(res2)
