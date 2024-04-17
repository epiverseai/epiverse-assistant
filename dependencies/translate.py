import re
import deep_translator


def replace_code_blocks_r(block1: str, block2: str) -> str:
    code_blocks_b1 = re.findall(r"```(.*?)```", block1, re.DOTALL)
    code_blocks_b2 = re.findall(r"```(.*?)```", block2, re.DOTALL)
    if len(code_blocks_b1) == len(code_blocks_b2):
        for i, block in enumerate(code_blocks_b1):
            block2 = block2.replace(code_blocks_b2[i], block, 1)
    else:
        return block1
    return block2


def replace_vars_blocks(block1: str, block2: str) -> str:
    code_blocks_b1 = re.findall(r" `&(.*?)`&", block1, re.DOTALL)
    code_blocks_b2 = re.findall(r" `&(.*?)`&", block2, re.DOTALL)
    if len(code_blocks_b1) == len(code_blocks_b2):
        for i, block in enumerate(code_blocks_b1):
            block2 = block2.replace(code_blocks_b2[i], block, 1)
    else:
        return block1
    return block2


def translate_en_es(text: str) -> str:
    try:
        text = text.replace("`", "`&").replace("`&`&`&", "```")
        translated = deep_translator.GoogleTranslator(source="en", target="es").translate(text)
        r1 = replace_code_blocks_r(text, translated)
        r2 = replace_vars_blocks(text, r1)
        r2 = r2.replace("`&", "`")
        return r2
    except:
        return text


def translate_es_en(text: str) -> str:
    try:
        translated = deep_translator.GoogleTranslator(source="es", target="en").translate(text)
        return translated
    except Exception:
        return text


if __name__ == "__main__":
    import inspect

    text1 = inspect.cleandoc(
        """
        The DataFrame to be used must first be read
        ```r
        data_event <- import_data_event(year=2019, name_event="Aggressions By Animals Potentially Transmitting Rabies")
        data_clean <- clean_data_sivigila(data_event)
        ```
        Now the standardization of geographic codes is done:
        ```r
        standardize_geo_cods(data_event=clean_data)
        ```
        Where `data_event` reads the data and the year of interest must be modified
    """
    )

    text2 = inspect.cleandoc(
        """
        A palindrome is a word, phrase, or sequence of characters that reads the same backward as forward. To check if a function is a palindrome in Python, we can use the following function:
        ```python
        def is_palindrome(func):
            return func == func[::-1]
        ```
        This function takes a function as an argument and checks if it is a palindrome by comparing the function with its reverse. If the function is a palindrome, the function will return `True`. Otherwise, it will return `False`.
    """
    )

    text3 = inspect.cleandoc(
        """
        A palindrome is a word, phrase, or sequence of characters that reads the same backward as forward. To check if a function is a palindrome in Python, we can use the following function:
        ```
        def is_palindrome(func):
            return func == func[::-1]
        ```
        This function takes a function as an argument and checks if it is a palindrome by comparing the function with its reverse. If the function is a palindrome, the function will return `True`. Otherwise, it will return `False`.
    """
    )

    text4 = inspect.cleandoc(
        """
        Sure, I'd be happy to help you with that! Here is a basic implementation of the bubble sort algorithm in Python:
        everybody loves bubble sort! 😊

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

    res1 = translate_en_es(text1)
    print(res1)
    print("\n\n===============")

    res2 = translate_en_es(text2)
    print(res2)
    print("\n\n===============")

    res3 = translate_en_es(text3)
    print(res3)
    print("\n\n===============")

    res4 = translate_en_es(text4)
    print(res4)
    print("\n\n===============")
