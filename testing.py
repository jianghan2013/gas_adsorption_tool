import unittest
from BJH_function import BJH_calculation
import numpy as np
import numpy.testing as npt
from BJH_function import test_isotherm
import matplotlib.pyplot as plt



class testing(unittest.TestCase):

    def test_get_gas_constant(self):

        N2 = BJH_calculation.get_gas_constant()
        Ar = BJH_calculation.get_gas_constant('Ar')
        npt.assert_almost_equal(N2['A'],9.53,decimal=3)
        npt.assert_almost_equal(N2['Vmol'], 34.67, decimal=3)
        npt.assert_almost_equal(Ar['A'], 10.44)
        npt.assert_almost_equal(Ar['Vmol'], 28)

    def test_insert_zero(self):
        b = BJH_calculation.insert_zero([1,2])
        npt.assert_array_almost_equal(b,[0,1,2])

    def test_thickness_Harkins_Jura(self):
        p_rel = 0.499167054965542
        t = 6.45502909433275
        npt.assert_almost_equal(BJH_calculation.thickness_Harkins_Jura(p_rel),t,decimal=6)

    def test_pressure_and_radius(self):
        const_N2 = BJH_calculation.get_gas_constant('N2')
        const_Ar = BJH_calculation.get_gas_constant('Ar')
        npt.assert_almost_equal(BJH_calculation.kelvin_radius(0.5,const_N2),13.74888374,decimal=7)
        npt.assert_almost_equal(BJH_calculation.kelvin_radius(0.5,const_Ar), 15.06173623, decimal=7)
        npt.assert_almost_equal(BJH_calculation.radius_to_pressure(13.74888374, const_N2), 0.5, decimal=7)
        npt.assert_almost_equal(BJH_calculation.radius_to_pressure(15.06173623, const_Ar), 0.5, decimal=7)

    def test_get_CSA_a(self):
        # case 1
        npt.assert_almost_equal(BJH_calculation.get_CSA_a(99,99,99,0,5,10),0)
        # case 2
        npt.assert_almost_equal(BJH_calculation.get_CSA_a(99,99,99,99,9,9),9999)
        # case 3
        npt.assert_almost_equal(BJH_calculation.get_CSA_a(1, np.array([0,1,2]), np.array([0,1,3]), 2, 3, 4)*10**(16),34.55751919,decimal=7)

    def test_restrict_isotherm(self):
        P = np.arange(0.1, 0.9, 0.1)
        Q = np.arange(0.2, 1.8, 0.2)
        P1, Q1 = BJH_calculation.restrict_isotherm(P, Q, 0.3, 0.8)
        npt.assert_array_almost_equal(P1,[0.3,  0.4,    0.5,    0.6,    0.7,    0.8])
        npt.assert_array_almost_equal(Q1, [0.6,  0.8,  1. ,  1.2,  1.4,  1.6])

    def test_get_porosity(self):
        pass
        #p, q, BJH_calculate_volume = test_isotherm.shale_3_14()
        # test for Ar
        #vpore_total, vpore_micro, vpore_meso = BJH_calculation.get_porosity(p, q, 'Ar')
        #npt.assert_almost_equal(vpore_total,q[-1]*28/22414.0)
        #npt.assert_almost_equal(vpore_micro,q[0] * 28 / 22414.0)
        #npt.assert_almost_equal(vpore_meso, (q[-1]-q[0]) * 28 / 22414.0)

    def test_BJH_main_function(self):
        #3_14 n2
        p_rels,q,my_volume = test_isotherm.shale_3_14()

        Davg,Vp,Vp_ccum,Vp_dlogD = BJH_calculation.BJH_main(p_rels,q,use_pressure=False)
        npt.assert_array_almost_equal(Vp,my_volume,decimal=8)

    def test_BJH_class(self):
        p, q, my_volume = test_isotherm.shale_3_14()
        # test 1
        my_BJH_unspline = BJH_calculation.BJH_method(p,q,use_pressure=False)
        my_BJH_unspline.do_BJH()
        npt.assert_array_almost_equal(my_BJH_unspline.Vp,my_volume,decimal=8)

        # test 2 using pressure
        my_BJH_spline = BJH_calculation.BJH_method(p, q, use_pressure=True)
        my_BJH_spline.do_BJH()


        npt.assert_almost_equal(my_BJH_spline.vpore_total, my_BJH_spline.q[-1] * 34.67 / 22414.0)
        npt.assert_almost_equal(my_BJH_spline.vpore_micro, my_BJH_spline.q[0] * 34.67 / 22414.0)
        npt.assert_almost_equal(my_BJH_spline.vpore_meso, (my_BJH_spline.q[-1] - my_BJH_spline.q[0]) * 34.67 / 22414.0)

        # compare porosity from unspline and spline
        npt.assert_almost_equal(my_BJH_unspline.vpore_total, my_BJH_spline.q[-1] * 34.67 / 22414.0)
        npt.assert_almost_equal(my_BJH_unspline.vpore_micro, my_BJH_spline.q[0] * 34.67 / 22414.0)
        npt.assert_almost_equal(my_BJH_unspline.vpore_meso, (my_BJH_spline.q[-1] - my_BJH_spline.q[0]) * 34.67 / 22414.0)

        # test 3 shale_3_14_spline
        p_s,q_s,Vp_s = test_isotherm.shale_3_14_spline()
        npt.assert_almost_equal(my_BJH_spline.vpore_total, q_s[-1]* 34.67 / 22414.0)
        #figure = plt.figure()
        #lg1,=plt.plot(my_BJH_1.p_raw,my_BJH_1.q_raw,'ro-',label='BJH raw')
        #lg2,=plt.plot(p,q,'k.',label='raw')
        #lg3,=plt.plot(my_BJH_1.p,my_BJH_1.q,'g^',label='BJH fixed')
        #plt.legend(handles = [lg1,lg2,lg3],loc=2 )
        #plt.show()
        #print(my_BJH_1.p)
        #print(my_BJH_1.q)
        #print(my_BJH_1.Vp)
        # test 2



        #assert np.assert_almost_equal
if __name__ == '__main__':
    unittest.main()