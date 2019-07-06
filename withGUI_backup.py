#IMPORT
from DoYOLO_backup import *
#IMPORT

class App:
    def __init__(self,title):
        self.home = Tk()
        self.title = title
        self.canvas = None
        self.canvas1 = None
        self.photo = None
        self.photo_choosenFile = None
        self.allWidth = self.home.winfo_screenwidth()
        self.allHeight = self.home.winfo_screenheight() - 100
        self.allFrameWidth = int(0.875*(self.allWidth/2))
        self.canvas_width = self.allFrameWidth#500
        self.canvas_scaling_coef = 1920/self.canvas_width
        self.canvas_height = int(1080/self.canvas_scaling_coef)#281 #int(1080/2.7429)#int(1080/self.canvas_scaling_coef)
        self.topAreaFrameHeight = int(0.7*self.allHeight)
        self.bottomAreaFrameHeight = int(0.3*self.allHeight)
        self.areaFramePaddingWidth = int(0.125*(self.allWidth/2))
        self.vid = None
        self.filename = None
        self.update_flag = None
        self.frame_counter = 0
        self.canvas_history = []

    def run(self):
        #inits apps
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

        #put widgets on Top Right Frame
        Label(self.frameTopRight,text="ROI").grid(row=0,column=0)
        self.canvas1 = Canvas(self.frameTopRight, width = int(self.canvas_width/3), height = int(self.canvas_height/3))
        self.canvas1.grid(row=0,column=1)
        ohio = Button(self.frameTopRight,text="Choose File")
        ohio.grid(row=0,column=2)
        #put widgets on Top Right Frame

        #put widgets on Bottom Left Frame
        self.detected_label = Label(self.frameBotLeftLeft,text="Plat Nomor Terdeteksi : ")
        self.detected_label.pack()

        self.the_labels = Label(self.frameBotLeftLeft,text="-",font=self.helv)
        self.the_labels.pack()

        self.frame_counter_toShow_label = Label(self.frameBotLeftLeft,text="Frame : ")
        self.frame_counter_toShow_label.pack()

        self.frame_counter_toShow = Label(self.frameBotLeftLeft,text='-',font=self.helv)
        self.frame_counter_toShow.pack()

        self.buttonPlay = Button(self.frameBotLeftMiddle,text="Play",width=8,height=3,font=self.helv,command=self.startVideo)
        self.buttonPlay.pack()

        Frame(self.frameBotLeftMiddle,height=10).pack()

        self.buttonStop = Button(self.frameBotLeftMiddle,text="Stop",width=8,height=3,command=self.stopVideo,font=self.helv)
        self.buttonStop.pack()

        self.buttonSave = Button(self.frameBotLeftRight,text="Save",width=8,height=3,font=self.helv,command=self.saveVideoFrame)
        self.buttonSave.pack()

        self.statusSave = Label(self.frameBotLeftRight,text="-")
        self.statusSave.pack()
        #put widgets on Bottom Left Frame

        # #update the frame from
        # self.update(delay = 20)
        # #update the frame from

        self.home.mainloop()

    def startVideo(self):
        if self.video_init == True:
            print('masuk if')
            self.video_init = False
            self.update_flag = True
            self.vid = VideoBackend(self.filename,self.canvas_width,self.canvas_height)
            self.buttonPlay['text'] = 'Pause'
            self.update(delay=10)
        elif self.update_flag == True:
            print('masuk elif true')
            self.pauseVideo()
            self.buttonPlay['text'] = 'Play'
        elif self.update_flag == False:
            print('masuk elif false')
            self.resumeVideo()
            self.buttonPlay['text'] = 'Pause'
            self.update(delay=10)

    def stopVideo(self):
        self.update_flag = False
        self.video_init = True
        self.buttonPlay['text'] = 'Play'
        self.frame_counter = 0
        self.frame_counter_toShow['text'] = self.frame_counter
        del self.vid

    def resumeVideo(self):
        self.update_flag = True

    def pauseVideo(self):
        self.update_flag = False

    def saveVideoFrame(self):
        temp = self.filename.split('/')
        temp1 = temp[-1]
        filename,ext = temp1.split('.')
        try:
            cv2.imwrite('resources/GUIresources/saved_frames/'+filename +'-'+ str(self.frame_counter) + '.jpg',self.frame)
            self.statusSave['text'] = 'Frame berhasil disimpan.'
            self.home.after(2000,self.clearStatusSave)
        except:
            self.statusSave['text'] = 'Frame gagal disimpan.'
            self.home.after(2000,self.clearStatusSave)

    def clearStatusSave(self):
        self.statusSave['text'] = '-'

    def choose_file(self):
        self.vid = None
        self.stopVideo()
        self.video_init = True
        self.vid = None
        self.filename = filedialog.askopenfilename(initialdir="./resources",title="Select File",filetypes=(("video files","*.MOV"),("all files","*")))
        the_vid = cv2.VideoCapture(self.filename)
        ret,frame = the_vid.read()
        frame_toShow = cv2.resize(frame,(self.canvas_width,self.canvas_height))
        frame_toShow = cv2.cvtColor(frame_toShow, cv2.COLOR_BGR2RGB)
        self.photo_choosenFile = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_toShow))
        self.canvas.create_image(0,0, image=self.photo_choosenFile,anchor = NW)

        print(self.filename)

    def update(self,delay = 10):
        try:
            self.ret, self.frame, self.frame_toShow,charas = self.vid.get_frame()
            self.frame_toShow = cv2.cvtColor(self.frame_toShow, cv2.COLOR_BGR2RGB)
            if self.ret:
                self.frame_counter+=1
                self.frame_counter_toShow['text'] = self.frame_counter
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame_toShow))
                self.canvas.create_image(0, 0, image = self.photo, anchor = NW)
                roi_nya, roi_nya_flag = self.vid.getROI()
                print("cinta")
                if roi_nya_flag == True:
                    print("cintaaa")
                    roi_nya = cv2.cvtColor(roi_nya,cv2.COLOR_BGR2RGB)
                    photo1 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(roi_nya))                    
                    cv2.imwrite('resources/GUIresources/history/his.jpg',roi_nya)
                    self.canvas1.create_image(0, 0, image = photo1, anchor = NW)
                self.the_labels['text'] = charas
                # if (self.frame_counter>=324 and self.frame_counter<=351) or (self.frame_counter>=452 and self.frame_counter<= 486) or (self.frame_counter>=575 and self.frame_counter<= 601) or (self.frame_counter>=634 and self.frame_counter <=650):
                # if (self.frame_counter>=95 and self.frame_counter<=123):
                # if (self.frame_counter>=161 and self.frame_counter<=177) or (self.frame_counter>=715 and self.frame_counter<= 733) or (self.frame_counter>=1066 and self.frame_counter<= 1082):
                #     cv2.imwrite("resources/GUIresources/saved/"+str(self.frame_counter)+".jpg",self.frame_toShow)

            if self.update_flag == True:
                self.home.after(delay,self.update)
        except:
            self.vid = None

    def putFrameTopLeft(self):
        Frame(self.home,width=self.areaFramePaddingWidth,height=self.topAreaFrameHeight).grid(row=0,column=0)
        self.frameTopLeft = Frame(self.home,width=self.allFrameWidth,height=self.topAreaFrameHeight)
        self.frameTopLeft.grid(row=0,column=1)

    def putFrameBotLeft(self):
        Frame(self.home,width=self.areaFramePaddingWidth,height=self.bottomAreaFrameHeight).grid(row=1,column=0)
        self.frameBotLeft = Frame(self.home,width=self.allFrameWidth,height=self.bottomAreaFrameHeight)
        self.frameBotLeft.grid(row=1,column=1)

        self.frameBotLeftLeft = Frame(self.frameBotLeft,width=int(0.4142*( self.allFrameWidth )),height=self.bottomAreaFrameHeight)
        self.frameBotLeftLeft.grid(row=0,column=0)

        Frame(self.frameBotLeft,width=int(0.0143*( self.allFrameWidth )),height=self.bottomAreaFrameHeight).grid(row=0,column=1)

        self.frameBotLeftMiddle = Frame(self.frameBotLeft,width=int(0.2714*( self.allFrameWidth )),height=self.bottomAreaFrameHeight)
        self.frameBotLeftMiddle.grid(row=0,column=2)

        Frame(self.frameBotLeft,width=int(0.0143*( self.allFrameWidth )),height=self.bottomAreaFrameHeight).grid(row=0,column=3)

        self.frameBotLeftRight = Frame(self.frameBotLeft,width=int(0.2857*( self.allFrameWidth )),height=self.bottomAreaFrameHeight)
        self.frameBotLeftRight.grid(row=0,column=4)

    def putFrameTopRight(self):
        Frame(self.home,width=self.areaFramePaddingWidth,height=self.topAreaFrameHeight,bg='yellow').grid(row=0,column=2)
        self.frameTopRight = Frame(self.home,width=self.allFrameWidth,height=self.topAreaFrameHeight,bg='green')
        self.frameTopRight.grid(row=0,column=3)

    def putFrameBotRight(self):
        Frame(self.home,width=self.areaFramePaddingWidth,height=self.bottomAreaFrameHeight,bg='red').grid(row=1,column=2)
        self.frameBotRight = Frame(self.home,width=self.allFrameWidth,height=self.bottomAreaFrameHeight,bg='blue')
        self.frameBotRight.grid(row=1,column=3)


class VideoBackend:
    def __init__(self,video_source,resize_width,resize_height):
        self.vid = cv2.VideoCapture(video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.roiCandidate = None
        self.roiFlag = False

    def get_frame(self):
        self.ret, self.frame = self.vid.read()
        self.frame_toNetwork = self.frame.copy()
        self.charas = ''
        
        coor1, coor2, pojok_kiri_atas, pojok_kanan_bawah = YOLO(self.frame_toNetwork)
        print("yolo done")
        if coor1==0:
            print("no detected roi")
            self.roiFlag = False
            self.charas = 'xx xxxx xx'
            pass
        else:
            self.roiFlag = True
            
            cv2.rectangle(self.frame_toNetwork,coor1,coor2,(0,255,0),2)
            self.roiCandidate = self.frame_toNetwork[pojok_kiri_atas[1]:pojok_kanan_bawah[1],pojok_kiri_atas[0]:pojok_kanan_bawah[0]]
            
            img_,eroded,the_charas,the_flag = segImg(self.roiCandidate,"")
            
            if len(the_charas)==0:
                self.charas = 'xx xxxx xx'
            else:
                temp_charas = ''.join(the_charas)
                temp = temp_charas.split('-')
                for ind in range(len(temp)):
                    if temp[ind] == '':
                        if ind==1:
                            self.charas+='xxxx '
                        elif ind==2:
                            self.charas+='xx'
                        elif ind==0:
                            self.charas+='xx '
                    else:
                        self.charas+=temp[ind]
                        if ind!=2:
                            self.charas+=' '
        
        self.frame_toShow = cv2.resize(self.frame_toNetwork,(self.resize_width,self.resize_height))
        if self.ret:
            return self.ret,self.frame,self.frame_toShow,self.charas
        else:
            return self.ret,None,None,self.charas
    
    def getROI(self):
        return self.roiCandidate,self.roiFlag

    def __del__(self):
        self.vid.release()

if __name__=="__main__":
    hehe = App("Tugas-Akhir")
    hehe.run()