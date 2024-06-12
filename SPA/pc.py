import os

class program_controller:
    def __init__(self, program_path) -> None:
        self.program_path = program_path
        
    def pre_process(self):
        pass
    
    def init_blocks(self):
        """To get a list of program block indexes (where we put the color command)
        self.blocks[block_num] = line_num
        the file is loaded in self.listOfLines, which would replace the original file
        """
        self.blocks = []
        #TBD:here use color to locate, this is a simplification, should use block rules
        fileHandler  =  open(self.program_path,  "r")
        self.listOfLines  =  fileHandler.readlines()
        fileHandler.close()
        for (line_num, line) in enumerate(self.listOfLines):
            if 'color([' in line:
                self.blocks.append(line_num)
        print("block list:", self.blocks)
    def init_blocks_plus(self):
        #for more generalize blocks initialization
        self.blocks = []
        fileHandler  =  open(self.program_path,  "r")
        self.listOfLines  =  fileHandler.readlines()
        fileHandler.close()
        #clean up colors
        self.listOfLines = [x for x in self.listOfLines if 'color' not in x]
        i=0
        while i < len(self.listOfLines):
            if 'gt label: ***' in self.listOfLines[i]:
                self.listOfLines.insert(i+1, 'color([0.1,0.1,0.1])\n')
                self.blocks.append(i+1)
                i +=2
            else:
                i +=1
        print("block list:", self.blocks)
        
   
    def add_caption_to_block(self, block_num, caption):
        line_num = self.blocks[block_num]
        if 'color([' not in self.listOfLines[line_num]:
                print("no color at line:", line_num)
                return
        newLine = self.listOfLines[line_num].replace('\n', f'//{caption}\n')
        self.listOfLines[line_num] = newLine
        
    def change_block_color(self, block_num, new_color):
        """change the color of the given block number

        Args:
            block_num (int)
            new_color (float,float,float)
        """
        line_num = self.blocks[block_num]
        if 'color([' not in self.listOfLines[line_num]:
            print("no color at line:", line_num)
            return
        newLine = 'color([{},{},{}])\n'.format(new_color[0],new_color[1],new_color[2])
        self.listOfLines[line_num] = newLine
        
    def save_res(self, target_path):
        file = open(target_path,'w')
        for line in self.listOfLines:
            file.write(line)
        file.close()
        
    def optimize_with_color(self, color_lt):
        #enumerate all color combination
        pass
        