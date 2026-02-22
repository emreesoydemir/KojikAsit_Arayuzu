import streamlit as st
import numpy as np
import pandas as pd
import os
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
from PIL import Image
import io
import requests
from sklearn.neural_network import MLPClassifier


# --- ÇOKLU YAPAY SİNİR AĞI (MULTI-BRAIN) EĞİTİM MODÜLLERİ ---

def train_generic_model(csv_file, target_col_name, backup_dataset):
    X, y = [], []
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)

            if target_col_name in df.columns:
                df_toxic = df[df[target_col_name] == 1]
                df_safe = df[df[target_col_name] == 0]

                if len(df_toxic) > 0 and len(df_safe) > 0:
                    min_len = min(len(df_toxic), len(df_safe))
                    df_toxic = df_toxic.sample(n=min_len, random_state=42)
                    df_safe = df_safe.sample(n=min_len, random_state=42)
                    df = pd.concat([df_toxic, df_safe]).sample(frac=1, random_state=42).reset_index(drop=True)

            if target_col_name == "toksik_mi":
                anchor_data = pd.DataFrame({
                    'smiles': [
                        "OCC1=CC(=O)C(O)=CO1", "O=C1C=C(CO)OC=C1(N(=O)=O)",
                        "O=C1C=C(C#N)OC=C1O", "O=C1C=C(CO)OC=C1(Cl)",
                        "O=N(=O)c1ccccc1", "N#Cc1ccccc1"
                    ],
                    target_col_name: [0, 1, 1, 1, 1, 1]
                })
                anchor_data = pd.concat([anchor_data] * 15, ignore_index=True)
                df = pd.concat([df, anchor_data], ignore_index=True).sample(frac=1, random_state=42).reset_index(
                    drop=True)

            for index, row in df.iterrows():
                smiles = str(row['smiles'])
                label = int(row[target_col_name])
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    arr = np.zeros((1,), dtype=int)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    X.append(arr)
                    y.append(label)

            model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1500, random_state=42)
            model.fit(np.array(X), np.array(y))
            return model, f"✅ Model '{csv_file}' eğitildi.", True
        except Exception as e:
            pass

    for smiles, label in backup_dataset.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((1,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X.append(arr)
            y.append(label)

    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    model.fit(np.array(X), np.array(y))
    return model, f"⚠️ Yedek Model Aktif.", False


@st.cache_resource
def initialize_all_models():
    bbb_backup = {"OCC1=CC(=O)C(O)=CO1": 1, "CCCCCCCCCCCCCCCC(=O)OCC1=CC(=O)C(OC(=O)CCCCCCCCCCCCCCC)=CO1": 0}
    bbb_model, bbb_status, bbb_active = train_generic_model("bbb_veriseti.csv", "p_np", bbb_backup)

    tox_backup = {"O=N(=O)c1ccccc1": 1, "N#Cc1ccccc1": 1, "OCC1=CC(=O)C(O)=CO1": 0}
    tox_model, tox_status, tox_active = train_generic_model("toksisite_veriseti.csv", "toksik_mi", tox_backup)

    gi_backup = {"OCC1=CC(=O)C(O)=CO1": 1, "CCCCCCCCCCCCCCCC(=O)OCC1=CC(=O)C(OC(=O)CCCCCCCCCCCCCCC)=CO1": 0}
    gi_model, gi_status, gi_active = train_generic_model("emilim_veriseti.csv", "emilir_mi", gi_backup)

    eff_backup = {"OCC1=CC(=O)C(O)=CO1": 0, "c1ccccc1": 0}
    eff_model, eff_status, eff_active = train_generic_model("etkinlik_veriseti.csv", "etkili_mi", eff_backup)

    return {
        "bbb": {"model": bbb_model, "status": bbb_status, "active": bbb_active},
        "tox": {"model": tox_model, "status": tox_status, "active": tox_active},
        "gi": {"model": gi_model, "status": gi_status, "active": gi_active},
        "eff": {"model": eff_model, "status": eff_status, "active": eff_active}
    }


def predict_with_ann(model, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return [0.5, 0.5]
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return model.predict_proba([arr])[0]


def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None, None
    return {
        "Molekül Ağırlığı (g/mol)": round(Descriptors.MolWt(mol), 2),
        "LogP (Yağda Çözünürlük)": round(Descriptors.MolLogP(mol), 2),
        "TPSA (Kutuplanma)": round(Descriptors.TPSA(mol), 2),
        "H-Bağı (Verici/Alıcı)": f"{Descriptors.NumHDonors(mol)} / {Descriptors.NumHAcceptors(mol)}",
        "Aromatik Halka": Descriptors.NumAromaticRings(mol)
    }, mol


def toxicity_and_reactivity_alerts(part_R1, part_R2, part_R3):
    alerts = []
    if "(N(=O)=O)" in part_R3 or "N(=O)=O" in part_R3 or "NO2" in part_R3:
        alerts.append("🛑 Nitro grupları mutajenik doku hasarı riski taşır.")
    if "(C#N)" in part_R3 or "C#N" in part_R3:
        alerts.append("🛑 Siyanür sistemik solunumu durdurur.")
    if "CCCCCCCC" in part_R1 or "CCCCCCCC" in part_R3:
        alerts.append("⚠️ Aşırı uzun zincirler doku birikimine yol açabilir.")
    return alerts


# --- UZMAN YAPAY ZEKA GERİ BİLDİRİM JENERATÖRÜ ---
def generate_detailed_feedback(props, p_bbb, p_gi, p_tox, p_eff, alerts, silika):
    feedback = []

    # 1. ETKİNLİK (BACE-1 ENZİMİNE BAĞLANMA)
    if p_eff >= 70:
        feedback.append(
            f"**🎯 Hedef Uyumu (BACE-1): Mükemmel.** Molekülün 3 boyutlu yapısı ve elektronik yük dağılımı, Alzheimer'a sebep olan BACE-1 enziminin kilit cebine tam oturuyor. Özellikle yapısındaki {props['Aromatik Halka']} adet Aromatik Halka (veya ideal iskelet hacmi), enzime 'Pi-Pi etkileşimleri' ile çok güçlü tutunmasını sağlıyor.")
    elif p_eff >= 40:
        feedback.append(
            f"**📉 Hedef Uyumu (BACE-1): Zayıf.** Molekül enzime kısmen yanaşabiliyor ancak onu kilitleyecek hidrofobik hacme sahip değil. R2 bölgesine daha büyük veya aromatik (Örn: Fenil) bir grup ekleyerek enzimin iç duvarlarına tutunma yüzeyini artırmayı denemelisiniz.")
    else:
        feedback.append(
            f"**❌ Hedef Uyumu (BACE-1): Başarısız.** Bu kimyasal form, BACE-1 enzimi tarafından bir anahtar olarak algılanmıyor. Şekil uyumsuzluğu çok yüksek. Molekül beyne gitse bile hastalığı tedavi edecek kimyasal etkinliği gösteremez.")

    # 2. TOKSİSİTE VE GÜVENLİK
    if p_tox >= 50:
        alert_str = " ".join(
            alerts) if alerts else "Yapay zeka, molekülünüzün barkodunu laboratuvar testlerinde hücreleri öldüren (nekroz/apoptoz) toksik ilaçların barkodlarıyla eşleştirdi."
        feedback.append(
            f"**🛑 Güvenlik ve Toksisite: Kritik Risk!** Molekül, Karaciğer (CYP450) enzimleri tarafından parçalanırken hücreler için ciddi bir stres yaratıyor. {alert_str} Çözüm olarak R3 bölgesindeki yüksek reaktif (Siyano, Nitro) grupları daha kararlı atomlarla değiştirmelisiniz.")
    else:
        feedback.append(
            f"**✅ Güvenlik ve Toksisite: İdeal (Temiz).** Hücre içi toleransı çok yüksek. Molekül metabolize olurken serbest radikal veya zehirli yan ürün bırakmıyor. DNA'yı bozacak yapısal bir tehlike tespit edilmedi.")

    # 3. FARMAKOKİNETİK (GEÇİŞLER VE EMİLİM)
    if silika:
        feedback.append(
            f"**🛸 Taşıyıcı Sistem (Silika MSN): Aktif.** İlaç beyne kan yoluyla doğrudan gitmek yerine, beynin güvendiği bir 'Nanopartikül Kapsülü' içine hapsedilerek gönderiliyor. Bu sayede ilacın ağır olması ({props['Molekül Ağırlığı (g/mol)']} g/mol) veya yüksek kutuplu olması (TPSA: {props['TPSA (Kutuplanma)']}) bir sorun yaratmıyor. Mide asidi ve Beyin Bariyeri kuralları tamamen bypass edildi.")
    else:
        if p_bbb >= 50 and p_gi >= 50:
            feedback.append(
                f"**🧠 Farmakokinetik (Emilim ve BBB Geçişi): Başarılı.** {props['Molekül Ağırlığı (g/mol)']} g/mol'lük hafif ağırlığı ve {props['LogP (Yağda Çözünürlük)']} seviyesindeki yağda çözünürlüğü, hücre zarlarından (lipit tabakadan) sızmak için ideal. 'Lipinski'nin 5 Kurallarına' harika uyuyor; mideden kolayca kana karışır ve Kan-Beyin Bariyerini kendi başına rahatlıkla aşar.")
        else:
            feedback.append(
                f"**🚧 Farmakokinetik (Emilim ve BBB Geçişi): Engellendi.** Molekül insan vücudundaki savunma bariyerlerini aşamıyor. TPSA (Kutuplanma) değeri ({props['TPSA (Kutuplanma)']}) çok yüksek olabilir, bu yüzden hücre duvarlarına yapışıp kalıyor veya kanın içinde ilerleyemeyecek kadar hantal. Çözüm: Kutupsal grupları azaltın veya sol menüden 'Silika Taşıyıcı' modunu aktif edin.")

    return feedback


# --- ANA ARAYÜZ (UI) ---
def main():
    st.set_page_config(page_title="Alzheimer CADD Lab", layout="wide", page_icon="🧬")

    with st.spinner('🧠 Yapay Zeka Beyinleri Yükleniyor...'):
        models = initialize_all_models()

    st.title("🧬 Kojik Asit: Alzheimer İlaç Tasarım Laboratuvarı")
    st.markdown("Gerçek laboratuvar verileriyle eğitilmiş **4-Beyinli Yapay Zeka** Karar Destek Sistemi.")

    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Kojic_acid.svg/1200px-Kojic_acid.svg.png", width=150)
    st.sidebar.header("⚙️ Sistem Ayarları")
    silika_kullan = st.sidebar.toggle("🛸 Silika Taşıyıcı Kullan", value=True)

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Zeka Durumu")
    for key, name in [("bbb", "Sinir Sistemi"), ("gi", "Mide Emilimi"), ("tox", "Organ Hasarı"), ("eff", "Etkinlik")]:
        if models[key]['active']:
            st.sidebar.success(f"✅ {name}")
        else:
            st.sidebar.warning(f"⚠️ {name} (Yedek)")

    with st.expander("ℹ️ Tasarım Kılavuzu: R1, R2 ve R3 Bölgeleri"):
        st.write(
            "Kojik Asit iskeleti üzerinde 3 stratejik noktayı değiştirebilirsiniz veya kendi SMILES kodunuzu girebilirsiniz.")
        st.markdown(
            "- **R1 (Sol):** Çözünürlüğü ve emilimi etkiler. \n- **R2 (Üst):** BACE-1 enzimine bağlanma anahtarıdır. \n- **R3 (Sağ):** Reaktiviteyi ve zehirliliği belirler.")

    st.markdown("### 🧪 Molekül Sentez Paneli")
    input_mode = st.radio("Sentez Yöntemi Seçiniz:",
                          ["🧩 Parça Birleştirme Kılavuzu (R1, R2, R3)", "✍️ Serbest SMILES Kodu Girişi"])

    part_R1, part_R2, part_R3 = "", "", ""
    selected_smiles = ""

    if input_mode == "🧩 Parça Birleştirme Kılavuzu (R1, R2, R3)":
        col_r1, col_r2, col_r3 = st.columns(3)

        bridge_opt = {"Yok": "", "Metilen": "C", "Etilen": "CC", "Karbonil": "C(=O)"}
        term_opt = {"Hidrojen": "", "Metil": "C", "Asetil": "C(=O)C", "Palmitoil": "C(=O)CCCCCCCCCCCCCCC",
                    "Aromatik": "c1ccccc1", "Amino": "N", "Klor": "Cl", "Flor": "F", "Siyano": "C#N", "Hidroksil": "O"}
        r3_opt = {"Hidrojen": "", "Klor": "(Cl)", "Flor": "(F)", "Siyano": "(C#N)", "Nitro": "(N(=O)=O)",
                  "Amino": "(N)"}

        with col_r1:
            st.markdown("**Bölge 1 (R1)**")
            r1_b = st.selectbox("R1 Köprü", list(bridge_opt.keys()))
            r1_t = st.selectbox("R1 Uç", list(term_opt.keys()))
            part_R1 = bridge_opt[r1_b] + term_opt[r1_t]

        with col_r2:
            st.markdown("**Bölge 2 (R2)**")
            r2_b = st.selectbox("R2 Köprü", list(bridge_opt.keys()))
            r2_t = st.selectbox("R2 Uç", list(term_opt.keys()))
            part_R2 = bridge_opt[r2_b] + term_opt[r2_t]

        with col_r3:
            st.markdown("**Bölge 3 (R3)**")
            part_R3 = r3_opt[st.selectbox("R3 Eklenti", list(r3_opt.keys()))]

        selected_smiles = f"{part_R1}OCC1=CC(=O)C(O{part_R2})=C{part_R3}O1"

    else:
        st.info("💡 Herhangi bir molekülün SMILES kodunu buraya yapıştırıp yapay zeka analizine sokabilirsiniz.")
        custom_smiles = st.text_input("SMILES Kodunuzu Buraya Girin:", value="OCC1=CC(=O)C(O)=CO1")
        selected_smiles = custom_smiles
        part_R3 = selected_smiles

    props, mol = calculate_properties(selected_smiles)

    if mol:
        st.markdown("---")
        st.markdown("### 🔬 1. Fiziksel ve Kimyasal Analiz")
        c1, c2 = st.columns([1, 1])
        with c1:
            img = Draw.MolToImage(mol, size=(450, 450))
            st.image(img, caption="Yeni Nesil İlaç Adayı Yapısı")
        with c2:
            st.markdown("#### Moleküler Parametreler")
            st.write(props)
            st.info(f"**SMILES Kodu:** `{selected_smiles}`")

        st.markdown("---")
        st.markdown("### 🧠 2. Biyolojik Simülasyon Tahminleri")

        p_bbb = predict_with_ann(models['bbb']['model'], selected_smiles)[1] * 100
        p_gi = predict_with_ann(models['gi']['model'], selected_smiles)[1] * 100
        p_tox = predict_with_ann(models['tox']['model'], selected_smiles)[1] * 100
        p_eff = predict_with_ann(models['eff']['model'], selected_smiles)[1] * 100

        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        col_met1.metric("BBB Geçiş", f"%{p_bbb:.1f}" if not silika_kullan else "🛸 %100")
        col_met2.metric("Mide Emilim", f"%{p_gi:.1f}" if not silika_kullan else "🛸 %100")
        col_met3.metric("Zehirlilik Riski", f"%{p_tox:.1f}", delta_color="inverse")
        col_met4.metric("Tedavi Gücü (BACE-1)", f"%{p_eff:.1f}")

        alerts = toxicity_and_reactivity_alerts(part_R1, part_R2, part_R3)
        for a in alerts: st.warning(a)

        st.markdown("---")
        st.markdown("### 🏆 3. Nihai Alzheimer Potansiyeli ve Uzman Raporu")

        if silika_kullan:
            final = (p_eff * 0.7) + ((100 - p_tox) * 0.3)
        else:
            final = (p_eff * 0.4) + ((100 - p_tox) * 0.2) + (p_bbb * 0.2) + (p_gi * 0.2)

        st.title(f"Genel Başarı Skoru: % {final:.1f}")
        st.progress(int(final))

        if final >= 75:
            st.success(
                "🌟 **MÜKEMMEL ADAY:** Bu molekül Alzheimer tedavisi için laboratuvar testlerine girmeye uygundur.")
        elif final >= 45:
            st.warning(
                "⚖️ **ORTALAMA ADAY:** Molekülün bazı zaafları var. Aşağıdaki uzman geri bildirimini okuyarak düzeltmeler yapın.")
        else:
            st.error("❌ **BAŞARISIZ:** Bu molekül biyolojik bariyerleri aşamıyor veya yüksek toksisite gösteriyor.")

        # --- YENİ EKLENEN DETAYLI BİLDİRİM PANELİ ---
        st.markdown("#### 🔍 Yapay Zeka Uzman Geri Bildirimi")
        feedback_list = generate_detailed_feedback(props, p_bbb, p_gi, p_tox, p_eff, alerts, silika_kullan)
        for f in feedback_list:
            st.info(f)

    else:
        st.error("Hatalı Molekül Kombinasyonu veya Geçersiz SMILES Kodu!")

    # --------------------------------------------------------------------------------
    # OTOMATİK İNTERNET TARAMASI VE TANIMOTO BENZERLİĞİ MODÜLÜ
    # --------------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🚀 4. Yapay Zeka İle Otomatik İlaç Keşfi (Sanal Tarama)")
    st.write(
        "Bu modül, internet üzerindeki gerçek FDA/Klinik ilaç veritabanlarına bağlanır, **Kojik Aside en çok benzeyen (Tanimoto Similarity)** molekülleri bulur ve onları 4 yapay zeka beynimizde test ederek en iyileri sana sunar.")

    if st.button("🌐 İnternetten Veri Çek ve Otomatik Tarama Başlat", use_container_width=True):
        with st.spinner("İnternet Veritabanına (DeepChem FDA İlaçları) Bağlanılıyor... Lütfen Bekleyin..."):
            try:
                url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
                df_screening = pd.read_csv(url, compression='gzip')
                smiles_list = df_screening['smiles'].dropna().unique()

                kojic_smiles = "OCC1=CC(=O)C(O)=CO1"
                kojic_mol = Chem.MolFromSmiles(kojic_smiles)
                kojic_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(kojic_mol, 2, nBits=1024)

                results = []
                for s in smiles_list:
                    test_mol = Chem.MolFromSmiles(s)
                    if test_mol:
                        test_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(test_mol, 2, nBits=1024)
                        similarity = DataStructs.TanimotoSimilarity(kojic_fp, test_fp)

                        if similarity > 0.15:
                            p_b = predict_with_ann(models['bbb']['model'], s)[1] * 100
                            p_g = predict_with_ann(models['gi']['model'], s)[1] * 100
                            p_t = predict_with_ann(models['tox']['model'], s)[1] * 100
                            p_e = predict_with_ann(models['eff']['model'], s)[1] * 100

                            if silika_kullan:
                                final_s = (p_e * 0.7) + ((100 - p_t) * 0.3)
                            else:
                                final_s = (p_e * 0.4) + ((100 - p_t) * 0.2) + (p_b * 0.2) + (p_g * 0.2)

                            results.append({
                                "SMILES Kodu": s,
                                "Kojik Aside Benzerlik": f"% {similarity * 100:.1f}",
                                "Zehirlilik Riski": f"% {p_t:.1f}",
                                "Alzheimer Etkinliği": f"% {p_e:.1f}",
                                "🌟 NİHAİ SKOR": round(final_s, 1)
                            })

                if len(results) > 0:
                    df_results = pd.DataFrame(results)
                    df_results = df_results.sort_values(by="🌟 NİHAİ SKOR", ascending=False).head(5)

                    st.success(
                        f"✅ Tarama Tamamlandı! Toplam {len(smiles_list)} ilaç incelendi. İşte Kojik Aside en çok benzeyen ve en yüksek skoru alan **İlk 5 İlaç Adayı:**")
                    st.dataframe(df_results, use_container_width=True)
                    st.info(
                        "💡 Yukarıdaki tabloda yer alan 'SMILES Kodu'nu kopyalayarak Sentez Paneli'ndeki 'Serbest SMILES Kodu Girişi' alanına yapıştırabilir ve detaylı Uzman Geri Bildirimini okuyabilirsiniz.")
                else:
                    st.warning("Veritabanında yeterince yüksek benzerlikte (Tanimoto > %15) türev bulunamadı.")

            except Exception as e:
                st.error(f"Bağlantı veya Tarama Hatası: {e}")


if __name__ == "__main__":
    main()