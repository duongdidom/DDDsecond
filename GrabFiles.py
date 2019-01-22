"""
in designated folder, find a bunch of pa2 files that being after 10:30pm.
find equivalent position file & 3 rc cons files
run those combination of files for 21% risk capital calculation
"""
import os
import glob
import datetime
from shutil import copy2
import sys
sys.path.insert(0, r"C:\Users\DDD\Documents\DuongDiDom1-GitHub\DDDthird")
import MainRC1

""" input """
parent_dir = r"D:\span\rc\out" # where output of live RiskCapital script is stored. E.g. D:\span\rc\out or C:\SPANfiles or C:\Users\DDD\Downloads\Test\201812
cutoff_time = "22:30:00"
date_start = "01/12/2018"   # beginning of capturing period
date_end = "04/12/2018"     # one day after end of capturing period
output_dir =  r"C:\Users\douglas.cao\Downloads"    # where files in execution list is copied into. E.g. C:\Users\douglas.cao\Downloads
#NOTE: output_dir should not be linked between local and remote desktop E.g. \\tsclient\C\SPANfiles\ 
""""""""""""

# 1. find a bunch of pa2 files that their modified time is after predefined cut off time
def Get_Timestamp():
    all_pa2 = glob.glob(parent_dir + "\\" + "*NZX.*" + "*.s.pa2")   # get list of pa2 files start with NZX... end with .s.pa2

    timestamp_list = []

    for pa2 in all_pa2: 
        mtime = os.path.getmtime(pa2)   # get modified time of pa2 file
        mtime = datetime.datetime.fromtimestamp(mtime)    # convert modified time from float to yyyy-mm-dd hh:mm:ss.ssss
        
        sameday_cutoff = mtime.replace(hour=cutoff_time.time().hour, minute=cutoff_time.time().minute, second=cutoff_time.time().minute, microsecond=0)   
        # from cut off time, find cut off time of the same day of the modified date & time
        sameday_cutoff_onehr_later = sameday_cutoff.replace(hour=sameday_cutoff.hour + 1, minute=sameday_cutoff.minute, second=sameday_cutoff.minute, microsecond=0)  

        if sameday_cutoff <= mtime <= sameday_cutoff_onehr_later and date_start <= mtime <= date_end: 
        # check if modified time of pa2 file is after cut off time
            timestamp = pa2[(len(parent_dir)+1):(len(parent_dir)+16)]     # extract time stamp from file name. From end of parent directory to the next 15 characters
            timestamp_list.append(timestamp)            # append time stamp to timestamp list

    return timestamp_list

# 2. find equivalent original position and rc_cons files at each time stamp
def Get_Files(timestamp_list):
    execution_list =[]
    # define an empty execution list. Will be the result of the function
    # purpose: store for each timestamp: pa2, position, & 3 rc cons files
    # header for execution list: ["date_time","pa2","position","rc_cons_Scan","rc_cons_Interm","rc_cons_Intercomm","rc_cons_House"]

    for timestamp in timestamp_list:    # loop through list
        # grab position and rc Cons files
            # glob() return result into a list
        posfile = glob.glob(parent_dir + "\\" + timestamp + "_SPANPOS*")
        rcCons_scan = glob.glob(parent_dir + "\\" + timestamp + "_rc_scan.csv")
        rcCons_intermonth = glob.glob(parent_dir + "\\" + timestamp + "_rc_intermonth.csv")
        rcCons_intercomm = glob.glob(parent_dir + "\\" + timestamp + "_rc_intercomm.csv")
        rcCons_house = glob.glob(parent_dir + "\\" + timestamp + "_house.csv")

        # check if those files exist
        if len(posfile) < 1:
            LOG.append("Position file for " + timestamp + " not found")
            posfile = "N/a"
        else:
            posfile = posfile[0][(len(parent_dir)+1):]
        if len(rcCons_scan) < 1:
            LOG.append("RC cons Scan file for " + timestamp + " not found")
            rcCons_scan = "N/a"
        else:
            rcCons_scan = rcCons_scan[0][(len(parent_dir)+1):]
        if len(rcCons_intermonth) < 1:
            LOG.append("RC cons Intermonth file for " + timestamp + " not found")
            rcCons_intermonth = "N/a"
        else:
            rcCons_intermonth = rcCons_intermonth[0][(len(parent_dir)+1):]
        if len(rcCons_intercomm) < 1:
            LOG.append("RC cons Intercomm file for " + timestamp + " not found")
            rcCons_intercomm = "N/a"
        else:
            rcCons_intercomm = rcCons_intercomm[0][(len(parent_dir)+1):]
        if len(rcCons_house) < 1:
            LOG.append("RC cons House file for " + timestamp + " not found")
            rcCons_house = "N/a"
        else:
            rcCons_house = rcCons_house[0][(len(parent_dir)+1):]

        # find pa2 filename again
        pa2 = glob.glob(parent_dir + "\\" + timestamp + "*NZX*" + "*.s.pa2")
        pa2 = pa2[0][(len(parent_dir)+1):]

        # append to execution list
        execution_list.append([timestamp,pa2, posfile,rcCons_scan, rcCons_intermonth,rcCons_intercomm, rcCons_house])

    return execution_list

# 3. copying files in execution list to temp folder
def Copy_2_out(execution_list):
    for eachdate in execution_list:
        for eachfile in eachdate[1:]: # skip the 1st item (date)
            copy2(parent_dir + "\\" + eachfile, output_dir)

# 4. define output and input file prior to running main RC calculation
def input_files(output_dir, out_timestamp, pa2_pa2, position_txt, cons_rc_scan_csv, cons_rc_intermonth_csv, cons_rc_intercomm_csv, cons_house_csv):
    ### skip finding the latest pa2 and position files
    ### skip copying pa2 and position file to another location
    pa2_pa2 = output_dir + pa2_pa2
    position_txt = output_dir + position_txt

    rc_intercomm_csv =  output_dir + cons_rc_intercomm_csv
    rc_intermonth_csv =  output_dir + cons_rc_intermonth_csv
    rc_scan_csv =  output_dir + cons_rc_scan_csv
    house_csv = output_dir + cons_house_csv

    ### 1.1. define a bunch of newly created files
    # modified files
    new_pa2 = output_dir + out_timestamp + r"_new.pa2"
    sum_position_txt = output_dir + out_timestamp + r"_sum.txt"
    sum_position_rc_txt = output_dir + out_timestamp + r"_sum_rc.txt"

    # newly created files
    whatif_xml = output_dir + out_timestamp + r"_whatif.xml"
    spanInstr_margin_txt = output_dir + out_timestamp + r"_spanInstr_margin.txt"  # span instruction for margin
    spanInstr_rc_txt = output_dir + out_timestamp + r"_spanInstr_rc.txt"  # span instruction for rc
    span_margin_spn = output_dir + out_timestamp + r"_span_margin.spn"    # spn files
    span_rc_spn = output_dir + out_timestamp + r"_span_rc.spn"
    pbreq_margin_csv = output_dir + out_timestamp + r"_pbreq_margin.csv"  # pbreq files
    pbreq_rc_csv = output_dir + out_timestamp + r"_pbreq_rc.csv"
    final_csv = output_dir + out_timestamp + r"_final.csv"

    return (pa2_pa2, position_txt, rc_intercomm_csv,rc_intermonth_csv, rc_scan_csv, house_csv, new_pa2, sum_position_txt,sum_position_rc_txt, whatif_xml, spanInstr_margin_txt, spanInstr_rc_txt, span_margin_spn, span_rc_spn, pbreq_margin_csv, pbreq_rc_csv, final_csv)

### MAIN ###
cutoff_time = datetime.datetime.strptime(cutoff_time,"%H:%M:%S") # convert cut off time from hh:mm:ss to yyyy-mm-dd hh:mm:ss
date_start = datetime.datetime.strptime(date_start,"%d/%m/%Y")  
date_end = datetime.datetime.strptime(date_end,"%d/%m/%Y")

LOG = []    # create an empty list for logging
LOG.append("start time: " + str(datetime.datetime.now()))   # insert start time to log list

timestamp_list = Get_Timestamp()    # get modified time from pa2 files that meet our predefined conditions
execution_list = Get_Files(timestamp_list)  # get equivalent lists of position files and 3 rc cons files for such timestamp
Copy_2_out(execution_list)  # copy those files to a new directory

# ask for user input if would like to run 21% RC calculation
if sys.version.startswith("3") and input("Calculate 21% RC ? ").lower() == "yes":   
# case python 3, use input()
    bCalc21percentRC = True
elif sys.version.startswith("2") and raw_input("Calculate 21% RC ? ").lower() == "yes":
# case python 3, use raw_input()
    bCalc21percentRC = True
else:
    bCalc21percentRC = False

if bCalc21percentRC == True:
    for eachdate in execution_list:
        print ("running: " + str(eachdate[0]))

        MainRC1.write_log(output_dir)

        # use function defined above to define input files and output files for mainRC calculation script
        (pa2_pa2, position_txt, rc_intercomm_csv,rc_intermonth_csv, rc_scan_csv, house_csv, new_pa2, sum_position_txt,sum_position_rc_txt, whatif_xml, spanInstr_margin_txt, spanInstr_rc_txt, span_margin_spn, span_rc_spn, pbreq_margin_csv, pbreq_rc_csv, final_csv) = input_files(
            output_dir + "\\" , # out put directory
            eachdate[0],    # output timestamp
            eachdate[1],    # pa2 
            eachdate[2],    # position 
            eachdate[3],    # rc scan
            eachdate[4],    # rc intermonth
            eachdate[5],    # rc intercomm
            eachdate[6]    # house
            )

        original_pa2 = MainRC1.read_pa2(pa2_pa2)

        (price_list, intermonth_list, intermonth_param_list, intercomm_list, instrument_list, option_list, deltascale_list, price_param_list, currency_list) = MainRC1.parse_pa2(original_pa2)

        (rc_scan_list, rc_intermonth_list, rc_intercomm_list) = MainRC1.read_rcparams (rc_scan_csv,rc_intermonth_csv, rc_intercomm_csv)

        (price_list, instrument_list) = MainRC1.calc_newscan(price_list, instrument_list, rc_scan_list, 0.21)

        (instrument_list, rc_intermonth_list, intermonth_list) = MainRC1.calc_newintermonth(instrument_list, intermonth_param_list, intermonth_list, rc_intermonth_list)

        intercomm_list = MainRC1.calc_newintercomm (rc_intercomm_list, intercomm_list)

        (new_intermonth_list, new_intercomm_list) = MainRC1.write_newinters(intermonth_list, intercomm_list, original_pa2)
        
        MainRC1.write_new_pa2(original_pa2,new_intermonth_list,new_intercomm_list,new_pa2)

        MainRC1.write_whatif(instrument_list, whatif_xml)

        position_list = MainRC1.read_position(position_txt)

        (sum_bpins_list, option_position_list, sum_bpacc_list) = MainRC1.parse_position(position_list)

        MainRC1.write_sum_position_txt(position_list,sum_bpins_list,sum_position_txt)

        MainRC1.write_sum_rc_position_txt(position_list,sum_bpins_list,sum_position_rc_txt)

        # calculate and get report for 
        ### margin
        MainRC1.margin_SPAN_instruction(pa2_pa2,sum_position_txt,span_margin_spn,spanInstr_margin_txt)

        MainRC1.call_SPAN(spanInstr_margin_txt)

        MainRC1.call_SPAN_report(span_margin_spn, pbreq_margin_csv)
    
        ### risk capital
        MainRC1.rc_SPAN_instruction(new_pa2,whatif_xml,sum_position_rc_txt,span_rc_spn,spanInstr_rc_txt)    # write txt instruction for SPAN calculation: risk capital
        MainRC1.call_SPAN(spanInstr_rc_txt)

        MainRC1.call_SPAN_report(span_rc_spn, pbreq_rc_csv)

        (option_position_list, sum_bpacc_list) = MainRC1.delta_adjust(
        option_list, deltascale_list, price_param_list,instrument_list,     # from pa2 file
        option_position_list, sum_bpacc_list                                # from position 
        )

        house_list = MainRC1.read_house(house_csv)

        pbreq_list = MainRC1.read_pbreqs(pbreq_margin_csv, pbreq_rc_csv)

        sum_bpacc_list = MainRC1.parse_rc(pbreq_list,sum_bpacc_list,house_list,currency_list)

        MainRC1.write_rc(sum_bpacc_list,final_csv)
            
LOG.append("finish time: " + str(datetime.datetime.now()))   # insert finish time to log list
for log in LOG: print (log)


