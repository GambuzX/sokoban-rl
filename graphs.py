import sys
from sokoban_utils.utils import *
from sokoban_utils.global_configs import GlobalConfigs

def main(argv):
    '''
    if len(argv) == 0:
        print("Program needs at least 1 argument!")
        print("Usage: python graphs.py file_names")
        return
    '''
    create_dir(GlobalConfigs.latex_graphs)  
    
    write_graph(GlobalConfigs.latex_graphs+"alpha.txt", "alpha", ["logs/sarsa_alpha.txt"])
    
    
def write_graph(file_name, var_name, data_files):
    file = open(file_name, "w")
    begin_graph(file, var_name)
    
    for data_file in data_files:
        newFile = open(data_file, "r")
        data_file_names = newFile.read().splitlines() 
        
        for data_file_name in data_file_names:
            
            file_data = open(data_file_name, "r")
            
            newFileLines = file_data.read().splitlines() 
        
            for i in range(int(newFileLines[0]) + 1, len(newFileLines)):
                addPlotOptions(file, "blue", "circle")
                
                line = newFileLines[i].split(",")
                addPair(file, line[0], line[1])
            
            addLegendEntry(file, data_file)
            file_data.close()
        
        newFile.close()
    
    end_graph(file)
    file.close()

def begin_graph(file, var_name):
    file.write("\\begin{tikzpicture\}\n")
    file.write("\\begin{axis\}[\n")
    file.write("\taxis lines = left,\n")
    file.write("\txlabel = {")
    file.write(var_name)
    file.write("},\n")
    file.write("\tylabel = {Reward\},\n")
    file.write("\tlegend style={at={(0.5,-0.2)},anchor=north},\n")
    file.write("\tenlarge x limits=-1,\n")
    file.write("\twidth=11cm,\n")
    file.write("\theight=10cm,\n")
    file.write("]\n")

def end_graph(file):
    file.write("\\end{axis\}\n")
    file.write("\\end{tikzpicture\}\n")

def addPlotOptions(file, color, mark):
    file.write("\\addplot[\n")
    file.write("\tcolor=")
    file.write(color)
    file.write(",\n")
    file.write("\tmark=")
    file.write(mark) 
    file.write(",\n")
    file.write("\t]\n")
    file.write("\tcoordinates {\n\t")

def addPair(file, x_value, y_value):
    file.write("(")
    file.write(x_value)
    file.write(",")
    file.write(y_value)
    file.write(")")

def addLegendEntry(file, name):
    file.write("};\n")
    file.write("\\addlegendentry{")
    file.write(name)
    file.write("}\n")


if __name__ == "__main__":
   main(sys.argv[1:])