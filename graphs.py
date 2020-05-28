import sys
from sokoban_utils.utils import *
from sokoban_utils.global_configs import GlobalConfigs

colors = [
    "red", "green", "blue", "cyan", "magenta", "yellow", 
    "black", "gray", "darkgray", "lightgray", 
    "brown", "lime", "olive", "orange", "pink", "purple", 
    "teal", "violet"
]

def main(argv):
    '''
    if len(argv) == 0:
        print("Program needs at least 1 argument!")
        print("Usage: python graphs.py file_names")
        return
    '''
    create_dir(GlobalConfigs.latex_graphs)  
    
    write_graph(GlobalConfigs.latex_graphs+"sarsa_alpha.txt", "alpha", ["logs/sarsa_alpha.txt"])
    write_graph(GlobalConfigs.latex_graphs+"qlearning_alpha.txt", "alpha", ["logs/qlearning_alpha.txt"])
    write_graph(GlobalConfigs.latex_graphs+"sarsa_gamma.txt", "gamma", ["logs/sarsa_gamma.txt"])
    write_graph(GlobalConfigs.latex_graphs+"qlearning_gamma.txt", "gamma", ["logs/qlearning_gamma.txt"])
    write_graph(GlobalConfigs.latex_graphs+"montecarlo_gamma.txt", "gamma", ["logs/montecarlo_gamma.txt"])
    write_graph(GlobalConfigs.latex_graphs+"sarsa_max_epsilon.txt", "max_epsilon", ["logs/sarsa_max_epsilon.txt"])
    write_graph(GlobalConfigs.latex_graphs+"qlearning_max_epsilon.txt", "max_epsilon", ["logs/qlearning_max_epsilon.txt"])
    write_graph(GlobalConfigs.latex_graphs+"montecarlo_max_epsilon.txt", "max_epsilon", ["logs/montecarlo_max_epsilon.txt"])
    
    
def write_graph(file_name, var_name, data_files):
    file = open(file_name, "w")
    begin_graph(file, var_name)
    
    for data_file in data_files:
        newFile = open(data_file, "r")
        data_file_names = newFile.read().splitlines() 
                    
        color_index = 0
                    
        for data_file_name in data_file_names:

            
            file_data = open(data_file_name, "r")
            
            newFileLines = file_data.read().splitlines() 
            
            addPlotOptions(file, colors[color_index%len(colors)], "halfcircle*")
        
            for i in range(1, int(newFileLines[0])):
                config_val = newFileLines[i].split(":")
                if config_val[0] == var_name:
                    val = round(float(config_val[1]), 2)    

        
            for i in range(int(newFileLines[0]) + 1, len(newFileLines)):
                line = newFileLines[i].split(",")
                addPair(file, line[0], line[1])
            
            addLegendEntry(file, var_name+"="+str(val))
            file_data.close()
            
            color_index += 1
        
        newFile.close()
    
    end_graph(file)
    file.close()

def begin_graph(file, var_name):
    file.write("\\begin{tikzpicture}\n")
    file.write("\\begin{axis}[\n")
    file.write("\taxis lines = left,\n")
    file.write("\txlabel = {")
    file.write("Number of Episodes") #var_name
    file.write("},\n")
    file.write("\tylabel = {Reward},\n")
    file.write("\tlegend columns = 2,\n")
    file.write("\tlegend style={at={(0.5,-0.2)},anchor=north},\n")
    file.write("\tenlarge x limits=-1,\n")
    file.write("\twidth=7cm,\n")
    file.write("\theight=7cm,\n")
    file.write("]\n")

def end_graph(file):
    file.write("\\end{axis}\n")
    file.write("\\end{tikzpicture}\n")

def addPlotOptions(file, color, mark):
    file.write("\\addplot[\n")
    file.write("\tonly marks,\n")
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