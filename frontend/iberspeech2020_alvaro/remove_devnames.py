import os
import numpy
import pandas
import re


SPKR = ['no-spkr', 'papa_francisco', 'isabel_moreno', 'angela_merkel', 'angeles_cortina', 'begona_villacis', 'jacob_petrus', 'paco_marin', 'jose_manuel_sanchez', 'mar_villalobos', 'felipe_vi', 'alberto_garzon', 'ignacio_aguado', 'isabel_garcia_tejerina', 'pol_monen', 'maria_casado', 'vladimir_putin', 'marta_marquez', 'quico_taronji', 'ramon_aranguena', 'pedro_sanchez', 'angi_fernandez', 'maria_de_nati', 'maria_marte', 'koldo_arrastia', 'meritxell_batet', 'nacho_marraco', 'manuel_huedo', 'angel_gabilondo', 'pablo_iglesias', 'san_morales', 'mariano_rajoy', 'albert_rivera', 'guillermo_campra', 'unai_sordo', 'inigo_errejon', 'pablo_echenique', 'david_solans', 'elsa_garcia_de_blas', 'antonio_resines', 'maria_gomez', 'eva_soriano', 'kyriakos_mitsotakis', 'carlos_e_cue', 'ivan_espinosa_delosmonteros', 'mavi_donate', 'elena_anaya', 'marina_castro', 'andrea_ventura', 'paco_churruca', 'jose_manuel_gonzalez_huesa', 'manuela_carmena', 'ximo_puig', 'oscar_casas', 'javier_ortega_smith', 'joan_mesquida', 'melisa_rodriguez', 'bieito_rubido', 'jaime_de_olano', 'juan_carlos_cuevas', 'silvia_vacas', 'mario_montero', 'teodoro_garcia_egea', 'sara_lozano', 'eduard_fernandez', 'manuel_marin', 'dolores_delgado', 'silvia_laplana', 'angela_garcia_romero', 'javier_casqueiro', 'ramon_colom', 'xabier_fortes', 'aitor_esteban', 'jesus_marana', 'fran_hervias', 'juan_carlos_girauta', 'federico_cardalus', 'rocio_monasterio', 'gerardo_olivares', 'monica_oltra', 'ingrid_rubio', 'antonio_roldan', 'noelia_vera', 'magdalena_valerio', 'julia_varela', 'juan_cruz', 'emiliano_garcia_page', 'cuca_gamarra', 'silvia_clemente', 'irene_montero', 'leo_rivera', 'anni_b', 'adria_collado', 'santigo_alveru', 'cristina_narbona', 'zahara', 'isabel_diaz_ayuso', 'agustin_almodovar', 'angela_chica', 'andrea_guasch', 'julia_creus', 'juanma_lopez_iturriaga', 'cristobal_montoro', 'gabriel_rufian', 'ignacio_camacho', 'arnaldo_otegi', 'jorge_ruiz', 'virginia_riezu', 'guillermo_lasheras', 'diego_dominguez', 'marcos_lopez', 'yonyi_arenas', 'cayetana_guillen_cuervo', 'leo_harlem', 'rafael_simancas', 'artur_mas', 'emmanuel_macron', 'monica_hernandez', 'sirun_demirjian', 'alfonso_guerra', 'juan_pablo_carpintero', 'carlos_areces', 'iolanda_marmol', 'angel_pons', 'adriana_lastra', 'ruben_tejerina', 'carolina_bona', 'jose_luis_abalos', 'guillermo_mariscal', 'juan_guaido', 'pablo_casado', 'maria_pedraza', 'miguel_angel_revilla', 'mike_pompeo', 'jose_maria_peridis', 'josep_borrel', 'fernando_grande_marlasca', 'jorge_lopez', 'santiago_abascal', 'ana_oramas', 'soraya_saenz_santamaria', 'cristina_olea', 'ruth_diaz', 'ione_belarra', 'almudena_ariza', 'vanesa_benedicto', 'mariola_cubells', 'alberto_casado', 'carmen_calvo', 'ana_pastor', 'leonor_mayor', 'javier_zapater', 'ines_arrimadas', 'pablo_bustinduy', 'eliseo_lizaran', 'chanel_terrero', 'jose_luis_martinez_almeida', 'theresa_may', 'jasmine_roldan', 'nativel_preciado', 'ricard_sabate', 'lucia_gil']

def _filter_special(text):
	"""
		Remove special characters from a str
	Args:
		text: str
	Return:
		lowercase str ascii-friendly
	"""
	text = text.lower()
	text = ''.join([i for i in text if not i.isdigit()])
	text = text.translate({ord(c): 'a' for c in 'áàäâ'})
	text = text.translate({ord(c): 'e' for c in 'éèëê'})
	text = text.translate({ord(c): 'i' for c in 'íìïî'})
	text = text.translate({ord(c): 'o' for c in 'óòöô'})
	text = text.translate({ord(c): 'u' for c in 'úùüû'})
	text = text.replace('ñ', 'n')
	text = text.replace('\n', '')
	text = text.replace('.', '')
	# text = re.sub(r"[^a-zA-Z0-9]+", ' ', text)
	return text


path = './datasets/IBERSPEECH20_nonull/validation.txt'

train = path.replace('validation', 'training')

train = pandas.read_csv(train, sep='\t', header=None)
# spkrs = sorted(list(set(train[1].values)))
train[1] = train[1].apply(lambda x: x.lower())
train[1] = train[1].apply(_filter_special)
trainspkrs = sorted(list(set(train[1].values)))
excluded = [x for x in trainspkrs if x not in SPKR]
train.drop(train[train[1].isin(excluded)].index, inplace = True)
train.to_csv(path.replace('validation', 'train2'), sep='\t', header=None, index=False)


val = pandas.read_csv(path, sep='\t', header=None)
val[1] = val[1].apply(lambda x: x.lower())
val[1] = val[1].apply(_filter_special)
valspkrs = sorted(list(set(val[1].values)))
excluded = [x for x in valspkrs if x not in SPKR]

val.drop(val[val[1].isin(excluded)].index, inplace = True)
val.to_csv(path.replace('validation', 'val2'), sep='\t', header=None, index=False)


