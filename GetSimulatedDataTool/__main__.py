import sys
from ROOT import TFile

def main():
    print("### Welcome to use GetSimulatedDataTool.")
    if len(sys.argv) != 3:
        print("### Usage: GetSimulatedDataTool <rootfile> <point_number>")
        print("### Usage: GetSimulatedDataTool out485_1.root 146")
        sys.exit(1)        
    root_file = sys.argv[1]
    point_number = int(sys.argv[2])
    print("root_file= ", root_file)
    print("point_number= ",point_number)

    file = TFile(root_file)
    #file.ls()
    t = file.Get("t")
    #t.Print()
    NEntries = t.GetEntries()
    print("NEntries = ",NEntries)
    SimulatedData = []
    for i in range(0, NEntries):
        t.GetEntry(i)
        if t.Point==point_number:
            if t.Turn < 10:
                print(" Turn= ",t.Turn, " point ",t.Point," time = ","%20.6f"%t.time," x = ", "%10.6f"%t.BE_X," y = ", "%10.6f"%t.BE_Y)
            SimulatedData.insert(i,[t.Turn,t.Point,t.time,t.BE_X,t.BE_Y])
    with open("SimulatedData.txt",'w') as file:
        for item in SimulatedData:
            file.write(str(item) + '\n')

    
if __name__ == "__main__":    
    main()
