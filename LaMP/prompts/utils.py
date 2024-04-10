def extract_strings_between_quotes(input_string):
    output_list = []
    inside_quotes = False
    current_string = ''
    
    for char in input_string:
        if char == '"' and not inside_quotes:
            inside_quotes = True
        elif char == '"' and inside_quotes:
            inside_quotes = False
            output_list.append(current_string)
            current_string = ''
        elif inside_quotes:
            current_string += char
    
    return output_list

def extract_after_article(input_string):
    article_index = input_string.find('article:')
    if article_index == -1:
        return None
    return input_string[article_index + len('article:'):].strip()

def extract_after_description(input_string):
    article_index = input_string.find('description:')
    if article_index == -1:
        return None
    return input_string[article_index + len('description:'):].strip()


def extract_after_review(input_string):
    article_index = input_string.find('review:')
    if article_index == -1:
        return None
    return input_string[article_index + len('review:'):].strip()

def extract_after_paper(input_string):
    article_index = input_string.find('paper:')
    if article_index == -1:
        return None
    return input_string[article_index + len('paper:'):].strip()

def extract_after_abstract(input_string):
    article_index = input_string.find('abstract:')
    if article_index == -1:
        return None
    return input_string[article_index + len('abstract:'):].strip()

def extract_after_colon(input_string):
    article_index = input_string.find(':')
    if article_index == -1:
        return None
    return input_string[article_index + len(':'):].strip()


def add_string_after_title(original_string, string_to_add):
    title_index = original_string.find("title")
    
    if title_index == -1:
        return original_string
    
    return original_string[:title_index+5] + ", and " + string_to_add + original_string[title_index+5:]

def batchify(lst, batch_size):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]