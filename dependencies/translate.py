import re
import deep_translator

def replace_code_blocks_r(block1: str, block2: str) -> str:
    code_blocks_b1 = re.findall(r"```r\n(.*?)\n```", block1, re.DOTALL)
    code_blocks_b2 = re.findall(r"```r\n(.*?)\n```", block2, re.DOTALL)
    if len(code_blocks_b1) == len(code_blocks_b2):
        for i, block in enumerate(code_blocks_b1):
            block2 = block2.replace(code_blocks_b2[i], block, 1)
    else:
        return block1
    return block2


def replace_vars_blocks(block1: str, block2: str) -> str:
    code_blocks_b1 = re.findall(r" `(.*?)` ", block1, re.DOTALL)
    code_blocks_b2 = re.findall(r" `(.*?)` ", block2, re.DOTALL)
    if len(code_blocks_b1) == len(code_blocks_b2):
        for i, block in enumerate(code_blocks_b1):
            block2 = block2.replace(code_blocks_b2[i], block, 1)
    else:
        return block1
    return block2


def translate_en_es(text: str) -> str:
    translated = deep_translator.GoogleTranslator(source="auto", target="es").translate(text)
    r1 = replace_code_blocks_r(text, translated)
    r2 = replace_vars_blocks(text, r1)
    return r2


def translate_es_en(text: str) -> str:
    translated = deep_translator.GoogleTranslator(source="auto", target="en").translate(text)
    return translated