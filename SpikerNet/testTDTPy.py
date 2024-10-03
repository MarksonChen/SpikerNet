from tdt import DSPProject, DSPError
try:
    # Load the circuit
    RZ2 = DSPProject()
    circuitRZ2 = RZ2.load_circuit('OstimTestPulse_Reinforce_RZ2.rcx', 'RZ2')
    circuitRZ2.start()
    print('Started RZ-2')

    RX7 = DSPProject()
    circuitRX7 = RX7.load_circuit('OstimTestPulse_Reinforce_RX7_Delay.rcx', 'RX7')
    circuitRX7.start()
    print('Started RX-7')

except DSPError:
    print("Error Starting TDT Circuits")