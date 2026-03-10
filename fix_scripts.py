#!/usr/bin/env python3  
  
f = open('d:/GIthub/Radiolaria-DINOv3/scripts/reproduce_paper.py', 'r', encoding='utf-8')  
content = f.read()  
f.close() 
  
content = content.replace(chr(34), '') 
  
f = open('d:/GIthub/Radiolaria-DINOv3/scripts/reproduce_paper.py', 'w', encoding='utf-8')  
f.write(content)  
f.close() 
  
print('Fixed reproduce_paper.py') 
