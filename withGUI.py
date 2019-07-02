#IMPORT
from ValidationPreprocess import *
#IMPORT


class App:
    def __init__(self,title):
        self.home = None
        self.title = title
        self.canvas = None
        self.photo = None
        self.photo_choosenFile = None
        self.allWidth = 1600
        self.allHeight = 1000
        self.canvas_width = 500
        self.canvas_height = 281
        self.vid = None
        self.filename = None
        self.update_flag = None

    def run(self):
        #inits apps
        self.home = Tk()
        self.home.geometry(str(self.allWidth)+'x'+str(self.allHeight))
        self.home.title(self.title)
        self.helv = font.Font(self.home,family='Helvetica', size=20, weight='bold')
        #init apps

        #put frames on apps
        self.putFrameTopLeft()
        self.putFrameBotLeft()
        self.putFrameTopRight()
        self.putFrameBotRight()
        #put frames on apps

        #put widgets on Top Left Frame
        self.buttonChooseFile = Button(self.frameTopLeft,text="Choose File",command=self.choose_file)
        self.buttonChooseFile.pack()
        self.canvas = Canvas(self.frameTopLeft, width = self.canvas_width, height = self.canvas_height)
        self.canvas.pack()

        #put widgets on Top Left Frame

        #put widgets on Bottom Left Frame
        self.detected_label = Label(self.frameBotLeftLeft,text="Plat Nomor Terdeteksi : ")
        self.detected_label.pack()

        self.the_labels = Label(self.frameBotLeftLeft,text="-",font=self.helv)
        self.the_labels.pack()

        self.buttonPlay = Button(self.frameBotLeftMiddle,text="Play",width=8,height=3,font=self.helv,command=self.startVideo)
        self.buttonPlay.pack()

        self.buttonStop = Button(self.frameBotLeftMiddle,text="Stop",command=self.stopVideo)
        self.buttonStop.pack()

        self.buttonSave = Button(self.frameBotLeftRight,text="Save",width=8,height=3,font=self.helv)
        self.buttonSave.pack()
        #put widgets on Bottom Left Frame

        # #update the frame from
        # self.update(delay = 20)
        # #update the frame from

        self.home.mainloop()

    def startVideo(self):
        self.update_flag = True
        self.vid = VideoBackend(self.filename)
        self.buttonPlay['text'] = 'Pause'
        self.update(delay=80)

    def stopVideo(self):
        self.update_flag = False
        self.buttonPlay['text'] = 'Play'
        del self.vid

    def choose_file(self):
        self.filename = filedialog.askopenfilename(initialdir="./resources",title="Select File",filetypes=(("video files","*.MOV"),("all files","*")))
        the_vid = cv2.VideoCapture(self.filename)
        ret,frame = the_vid.read()
        frame_toShow = cv2.resize(frame,(500,281))
        self.photo_choosenFile = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_toShow))
        self.canvas.create_image(0,0, image=self.photo_choosenFile,anchor = NW)

        print(self.filename)

    def update(self,delay = 10):
        try:
            self.ret, self.frame, self.frame_toShow = self.vid.get_frame()
            if self.ret:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame_toShow))
                self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

            if self.update_flag == True:
                self.home.after(delay,self.update)
        except:
            self.vid = None

    def putFrameTopLeft(self):
        Frame(self.home,width=100,height=700).grid(row=0,column=0)
        self.frameTopLeft = Frame(self.home,width=700,height=700)
        self.frameTopLeft.grid(row=0,column=1)

    def putFrameBotLeft(self):
        Frame(self.home,width=100,height=300).grid(row=0,column=0)
        self.frameBotLeft = Frame(self.home,width=700,height=300)
        self.frameBotLeft.grid(row=1,column=1)

        self.frameBotLeftLeft = Frame(self.frameBotLeft,width=290,height=300)
        self.frameBotLeftLeft.grid(row=0,column=0)

        Frame(self.frameBotLeft,width=10,height=300).grid(row=0,column=1)

        self.frameBotLeftMiddle = Frame(self.frameBotLeft,width=190,height=300)
        self.frameBotLeftMiddle.grid(row=0,column=2)

        Frame(self.frameBotLeft,width=10,height=10).grid(row=0,column=3)

        self.frameBotLeftRight = Frame(self.frameBotLeft,width=200,height=300)
        self.frameBotLeftRight.grid(row=0,column=4)

    def putFrameTopRight(self):
        self.frameTopRight = Frame(self.home,width=700,height=700)
        self.frameTopRight.grid(row=0,column=2)

    def putFrameBotRight(self):
        Frame(self.home,width=100,height=300).grid(row=0,column=2)
        self.frameBotRight = Frame(self.home,width=700,height=300)
        self.frameBotRight.grid(row=1,column=3)


class VideoBackend:
    def __init__(self,video_source):
        self.vid = cv2.VideoCapture(video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        self.ret, self.frame = self.vid.read()
        self.frame_toShow = cv2.resize(self.frame,(500,281))
        if self.ret:
            return self.ret,self.frame,self.frame_toShow
        else:
            return self.ret,None,None

    def __del__(self):
        self.vid.release()

if __name__=="__main__":
    args = sys.argv
    title_ind = args.index("--title") + 1
    title = args[title_ind]
    hehe = App(title)
    hehe.run()