import inspect


def get_answer(text: str):
    parts = text.split("Response:")
    if len(parts) > 1:
        return inspect.cleandoc("".join(parts[1:]).strip())
    else:
        return text


if __name__ == "__main__":
    import inspect

    text1 = inspect.cleandoc(
        """
        [INST] <<<SYS>>> You are an expert in any topic, please response the following  <<<SYS>>> Program me a bubble sort algorithm in python [/INST] Response: Sure, I'd be happy to help you with that! Here is a basic implementation of the bubble sort algorithm in Python:
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

    res1 = get_answer(text1)
    print(res1)
