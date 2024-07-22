# Input variables used in the main.py to train the model
input_var = ["gest_age", "bw", "day_since_birth", 'RxDay',
            # 'RxDayShifted', 'shiftedBy',
            "TodaysWeight", "TPNHours", "max_chole_TPNEHR",
            "Alb_lab_value","Ca_lab_value","Cl_lab_value","Glu_lab_value","Na_lab_value", 'BUN_lab_value',
            'Cr_lab_value','Tri_lab_value','ALKP_lab_value','CaI_lab_value',
            'CO2_lab_value', 'PO4_lab_value', 'K_lab_value', 'Mg_lab_value', 'AST_lab_value',
            "ALT_lab_value",
             "EnteralDose", "FluidDose", "VTBI", "InfusionRate",
             "FatInfusionRate",
            "ProtocolName_NEONATAL", "ProtocolName_PEDIATRIC",
            "LineID_1", "LineID_2",
            'gender_concept_id_0', 'gender_concept_id_8507', 'gender_concept_id_8532',  #<<<<<<<<<<<<< proccess it
            'race_concept_id_0', 'race_concept_id_8515', 'race_concept_id_8516',
            'race_concept_id_8527', 'race_concept_id_8557', 'race_concept_id_8657',
            'FatProduct_SMOFlipid 20%', 'FatProduct_Intralipid 20%', 'FatProduct_Omegaven 10%']


# Output variables
vars_3 = ['FatDose', 'AADose', 'DexDose', 'Acetate', 'Calcium', 'Copper', 'Famotidine', 'Heparin', 'Levocar',
          'Magnesium', 'MVIDose', 'Phosphate', 'Potassium', 'Ranitidine', 'Selenium', 'Sodium', 'Zinc']



