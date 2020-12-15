with open("/code/demo/result.txt", 'r') as f:
    text = f.read()
    text_list = text.split('/')
    
    with open("/code/demo/result_2.txt", 'w') as f_2: 
        for i in range(len(text_list)):
            f_2.write(f"{text_list[i]}\n")