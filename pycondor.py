import pyemu





if False:
    pyemu.os_utils.run("pestpp-swp {0} /h :{1} 1>{2} 2>{3}". \
                       format(self.pst.filename, self.port, master_stdout, master_stderr))


    pyemu.utils.start_slaves(self.slave_dir,"pestpp-swp",self.pst.filename,
                                         self.num_slaves,slave_root='..',port=self.port)


def generate_submit_file():
    pass